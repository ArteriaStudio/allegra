// LLM操作（liblm）
#include	"pch.h"
#include	<llama.h>
#include	<json-schema-to-grammar.h>
#include	<nlohmann/json.hpp>
#include	<misc/libtx/txText.h>
#include	<libux/UxTools.h>
#include	"liblm/liblm.h"




CLLMContext::CLLMContext()
{
	m_pContext = nullptr;
	m_pLLM = nullptr;
	m_pListener = nullptr;
}

CLLMContext::~CLLMContext()
{
	m_pListener = nullptr;
	m_pLLM = nullptr;
	m_pContext = nullptr;
}

//　
int
CLLMContext::CreateContext(CCtrlLLM & pLLM, ILLMListener * pListener)
{
	//　コンテキストを作成
	llama_context_params	ctx_params = ::llama_context_default_params();
	ctx_params.n_ctx = 2048; // コンテキストサイズ
	m_pContext = llama_init_from_model(pLLM.GetInterface(), ctx_params);
	if (!m_pContext) {
		return((int)ECtrlLM::FailedCreateModel);
	}

	/*
	//　文法サンプラをロード
	std::string		pSchema;
	nlohmann::json	pSchemaJSON = nlohmann::json::parse(pSchema);
	m_pGrammar = json_schema_to_grammar(pSchemaJSON);
	*/


	m_pListener = pListener;
	m_pLLM = &pLLM;

	return(0);
}

//　
void
CLLMContext::DeleteContext()
{
	m_pListener = nullptr;
	m_pLLM = nullptr;
	::llama_free(m_pContext);
	m_pContext = nullptr;

	return;
}

//　
int
CLLMContext::Sample(VChatMessagesA & pMessagesA, u8stringstream & pStream)
{
	VChatMessages	pMessages;

	for (const auto & pMessageA : pMessagesA) {
		TChatMessage	pMessage;

		std::u8string	pOutText;
		::SJIStoUTF8(pMessageA.pRole.size(), pMessageA.pRole.c_str(), pOutText);
		size_t	nUTF8Text = pOutText.length();
		pMessage.pRole.resize(nUTF8Text);
		memcpy(pMessage.pRole.data(), pOutText.data(), nUTF8Text);

		::SJIStoUTF8(pMessageA.pContent.size(), pMessageA.pContent.c_str(), pOutText);
		nUTF8Text = pOutText.length();
		pMessage.pContent.resize(nUTF8Text);
		memcpy(pMessage.pContent.data(), pOutText.data(), nUTF8Text);

		pMessages.push_back(pMessage);
	}

	return(CLLMContext::Sample(pMessages, pStream));
}

//　
int
CLLMContext::Sample(VChatMessagesW & pMessagesW, u8stringstream & pStream)
{
	VChatMessages	pMessages;

	for (const auto & pMessageW : pMessagesW) {
		TChatMessage	pMessage;

		std::u8string	pOutText;
		::UCS2toUTF8(pMessageW.pRole.size(), pMessageW.pRole.c_str(), pOutText);
		size_t	nUTF8Text = pOutText.length();
		pMessage.pRole.resize(nUTF8Text);
		memcpy(pMessage.pRole.data(), pOutText.data(), nUTF8Text);

		::UCS2toUTF8(pMessageW.pContent.size(), pMessageW.pContent.c_str(), pOutText);
		nUTF8Text = pOutText.length();
		pMessage.pContent.resize(nUTF8Text);
		memcpy(pMessage.pContent.data(), pOutText.data(), nUTF8Text);

		pMessages.push_back(pMessage);
	}

	return(CLLMContext::Sample(pMessages, pStream));
}

//　
int
CLLMContext::Sample(VChatMessages & pMessages, u8stringstream & pStream)
{
	std::u8string		pPromptText;
	CChatTemplate::Apply(m_pLLM, pMessages, pPromptText);

	const char *	pPrompt = (const char *)pPromptText.data();
	int32_t 		nPrompt = (int32_t)pPromptText.length();

	//　プロンプト文字列をトークンに変換
	assert(m_pLLM);
	const llama_vocab * 	vocab = llama_model_get_vocab(m_pLLM->GetInterface());

	std::vector<llama_token> tokens(nPrompt + 16);
	int n_tokens = llama_tokenize(vocab, pPrompt, nPrompt, tokens.data(), (int32_t)tokens.size(), true, true);
	if (n_tokens < 0) {
		tokens.resize(-n_tokens);
		n_tokens = llama_tokenize(vocab, pPrompt, nPrompt, tokens.data(), (int32_t)tokens.size(), true, true);
	}
	tokens.resize(n_tokens);

	// 5. プロンプトの評価 (KVキャッシュへの格納)
	llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
	if (llama_decode(m_pContext, batch) != 0) {
		return 1;
	}

	// 6. サンプラーの初期化
	auto pSampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
	llama_sampler_chain_add(pSampler, llama_sampler_init_temp(0.7f));
	llama_sampler_chain_add(pSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

	llama_sampler * 	pGrammerSampler = llama_sampler_init_grammar(vocab, m_pGrammar.c_str(), "root");
	if (pGrammerSampler) {
		llama_sampler_chain_add(pSampler, pGrammerSampler);
	}

	// 7. テキスト生成ループ
//	u8stringstream		pStream;
	int max_tokens = 64000;
	int i;
	for (i = 0; i < max_tokens; ++i) {
		// 次のトークンをサンプリング
		llama_token new_token = llama_sampler_sample(pSampler, m_pContext, -1);

		// 終了トークン (EOS) か判定
		if (llama_vocab_is_eog(vocab, new_token)) {
			break;
		}

		// トークンを文字列に変換して出力
		char	buf[128] = {};
		auto n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
		//auto n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
		if (n > 0) {
			OnResponse(pStream, n, reinterpret_cast<const char8_t*>(buf));
		}

		//　サンプラの状態を更新
		llama_sampler_accept(pSampler, new_token);

		// 生成されたトークンをコンテキストに入力して次を予測
		batch = llama_batch_get_one(&new_token, 1);
		if (llama_decode(m_pContext, batch) != 0) {
			break;
		}
	}
	if (m_pListener) {
		auto p = pStream.str();
		auto n = pStream.str().length();
		m_pListener->OnResponse((const int32_t)n, p.c_str());
	}

	return(0);
}

//　
void
CLLMContext::OnResponse(u8stringstream & pStream, const int32_t nText, const char8_t * pText)
{
	std::u8string	pToken = pText;

	if (pToken.compare(u8"\n\n") == 0) {
		//　改行
		pStream << u8"\n";
	} else {
		pStream << pText;
	}

	return;
}
