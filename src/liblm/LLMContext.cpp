// LLM操作（liblm）
#include	"pch.h"
#include	<sstream>
#include	<llama.h>
#ifdef		ENABLE_GBNF_SCHEMA
#pragma 	warning(push)
#pragma 	warning(disable: 4305)
#pragma 	warning(disable: 4244)
#include	<common/common.h>
#pragma 	warning(pop)
#include	<common/sampling.h>
#include	<common/json-schema-to-grammar.h>
#endif	//	ENABLE_GBNF_SCHEMA
//#include	<nlohmann/json.hpp>
#include	<misc/libtx/txText.h>
#include	<libux/UxTools.h>
#include	<liblm/liblm.h>




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

#ifdef		ENABLE_GBNF_SCHEMA
	//　文法サンプラをロード
/*
	std::string 	pSchema = R"({
		"type": "object",
			"properties": {
			"name": { "type": "string" },
			"age": { "type": "integer" },
			"skills": {
			"type": "array",
			"items": { "type": "string" }
			}
		},
			"required": ["name", "age"]
	})";
*/
	std::string 	pSchema_JSON = R"({"id": 1,"name": "Ronova","is_active": true})";


	pSchema_JSON =R"({
	"type": "array",
	"items": {
		"type": "object",
		"properties": {
			"name": {
				"type": "string",
				"minLength": 1,
				"maxLength": 100
			},
			"age": {
				"type": "integer",
				"minimum": 0,
				"maximum": 150
			}
		},
		"required": ["name", "age"],
		"additionalProperties": false
	},
	"minItems": 10,
	"maxItems": 100
  })";



	pSchema_JSON = R"({"type": "integer","minimum": 1})";
	pSchema_JSON = R"({
	"type": "object",
	"properties": {
		"value": {
			"type": "integer"
		}
	},
	"required": ["value"]
})";


	try {

		// 空オブジェクトの場合
//		auto j1 = common_json::parse(R"({})");
		// 空配列の場合
//		auto j2 = common_json::parse(R"([])");
		//	nlohmann::json	pSchemaJSON = nlohmann::json::parse(pSchema);
		//	auto pSchemaJSON = json_schema_to_grammar(pSchema);
		//common_json	pSchema = pSchema_JSON;
		//common_json pSchema = common_json::parse(reinterpret_cast<const char *>(pSchema_JSON.c_str()));

		common_json 	pSchema = common_json::parse(pSchema_JSON);
		m_pGrammar = ::json_schema_to_grammar(pSchema, true);
	}
	catch (const std::exception& e) {
		// JSON パースエラーや内部例外を補獲
		std::stringstream	pOut;
		pOut << "Caught exception: " << e.what() << std::endl;
		::OutputDebugStringA(pOut.str().c_str());
	}
	catch (...) {
		std::stringstream	pOut;
		pOut << "Caught unknown exception" << std::endl;
		::OutputDebugStringA(pOut.str().c_str());
	}
#endif	//	ENABLE_GBNF_SCHEMA


	m_pListener = pListener;
	m_pLLM = &pLLM;

	return(0);
}

//　
void
CLLMContext::DeleteContext()
{
//	::llama_state_save_file(m_pContext, "",)


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

//　LLM クエリー
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

	// 6. サンプラーの初期化
	auto pSampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
	llama_sampler_chain_add(pSampler, llama_sampler_init_temp(0.7f));
	llama_sampler_chain_add(pSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

#ifdef		ENABLE_GBNF_SCHEMA
	llama_sampler * 	pGrammerSampler = llama_sampler_init_grammar(vocab, m_pGrammar.c_str(), "root");
	if (pGrammerSampler) {
		llama_sampler_chain_add(pSampler, pGrammerSampler);
	}
#endif	//	ENABLE_GBNF_SCHEMA

	// 5. プロンプトの評価 (KVキャッシュへの格納)
	llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
	if (llama_decode(m_pContext, batch) != 0) {
		return 1;
	}

	/*
	//　サンプラの状態を更新
	for (int i = 0; i < n_tokens; ++i) {
		llama_sampler_accept(pSampler, tokens[i]);
	}
	*/

	// 7. テキスト生成ループ
//	u8stringstream		pStream;
	int max_tokens = 64000;
	int i;
	for (i = 0; i < max_tokens; ++i) {
		// 次のトークンをサンプリング
		try {
			llama_token 	new_token = llama_sampler_sample(pSampler, m_pContext, -1);

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
		catch (const std::exception& e) {
			// JSON パースエラーや内部例外を補獲
			std::stringstream	pOut;
			pOut << "Caught exception: " << e.what() << std::endl;
			::OutputDebugStringA(pOut.str().c_str());
		}
		catch (...) {
			std::stringstream	pOut;
			pOut << "Caught unknown exception" << std::endl;
			::OutputDebugStringA(pOut.str().c_str());
		}
	}
	llama_decode(p)


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
