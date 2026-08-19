// LLM操作（liblm）
#include	"pch.h"
#include	<misc/libtx/txText.h>
#include	<libux/UxTools.h>
#include	"liblm/liblm.h"

//　関数プロトタイプ
static void 	log_callback_null(ggml_log_level level, const char* text, void* user_data);

//　インポートライブラリ
#pragma comment(lib, "ggml.lib")
#pragma comment(lib, "ggml-base.lib")
#pragma comment(lib, "ggml-cpu.lib")
#pragma comment(lib, "ggml-vulkan.lib")
#pragma comment(lib, "D:/App/Vulkan/lib/vulkan-1.lib")
#pragma comment(lib, "llama.lib")

CLlama::CLlama()
{
}

CLlama::~CLlama()
{
}

//　llama.cppのバックエンドを初期化
int
CLlama::Initialize()
{
	::llama_log_set(log_callback_null, nullptr);

	::llama_backend_init();


	return(0);
}

//　llama.cppのバックエンドを後始末
void
CLlama::Finalize()
{
	::llama_backend_free();

	return;
}

static void
log_callback_null(ggml_log_level level, const char* text, void* user_data) {
	(void)level;
	(void)text;
	(void)user_data;
}


CCtrlLLM::CCtrlLLM()
{
	m_pModel = nullptr;
}

CCtrlLLM::~CCtrlLLM()
{
	m_pModel = nullptr;
}

//　
int
CCtrlLLM::Create()
{
	//　モデルをロード
	std::string model_path = "D:\\Home\\Assets\\Models\\LLM\\gemma-4-gguf-gemma-4-e4b-it-qat-q4_0-gguf-v2\\gemma-4-E4B_q4_0-it.gguf"; // 使用するGGUFモデルのパス
	llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = 99; // GPUにオフロードするレイヤー数 (0でCPUのみ)

	m_pModel = ::llama_model_load_from_file(model_path.c_str(), model_params);
	if (!m_pModel) {
		return((int)ECtrlLM::FailedLoadModel);
	}

	return(0);
}

//　
void
CCtrlLLM::Delete()
{
	::llama_model_free(m_pModel);
	m_pModel = nullptr;

	return;
}

llama_model *
CCtrlLLM::GetInterface(void)
{
	assert(m_pModel);
	return(m_pModel);
}


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

CChatTemplate::CChatTemplate()
{
}

CChatTemplate::~CChatTemplate()
{
}


//　Generate by Gemini3.7 Flash
int32_t
CChatTemplate::ApplyGemmaFormat(VChatMessages & pMessages, std::string & pPrompt, bool add_generation_prompt)
{
	for (const auto & msg : pMessages) {
		auto role_tag = msg.pRole;
		// assistant を model に正規化
		if (role_tag == "assistant") {
			role_tag = "model";
		}

		pPrompt += "<|turn>" + role_tag + "\n";
		pPrompt += msg.pContent;
		pPrompt += "<turn|>\n";
	}

	if (add_generation_prompt) {
		pPrompt += "<|turn>model\n";
	}

	return((int32_t)pPrompt.length());
}

//　
int32_t
CChatTemplate::Apply(CCtrlLLM * pLLM, VChatMessages & pMessages, std::string & pPrompt, bool add_generation_prompt)
{
	std::string 	pText;

	pText.resize(256);

	auto n0 = llama_model_desc(pLLM->GetInterface(), pText.data(), pText.capacity());
	std::string 	pDesc(pText.data(), n0);

	auto n1 = llama_model_meta_val_str(pLLM->GetInterface(), "general.architecture", pText.data(), pText.capacity());
	std::string 	pArch(pText.data(), n1);

	auto n2 = llama_model_meta_val_str(pLLM->GetInterface(), "general.name", pText.data(), pText.capacity());
	std::string 	pName(pText.data(), n2);

//	auto n3 = llama_model_meta_val_str(pLLM->GetInterface(), "general.basename", pText.data(), pText.capacity());
//	pDesc = pText;

	int32_t 	nPrompt = 0;
	if (pArch.compare("gemma4") == 0) {
		nPrompt = ApplyGemmaFormat(pMessages, pPrompt, add_generation_prompt);
		/*
		auto pText = std::format(L"{}\n", pPrompt.c_str());
		::OutputDebugStringW(pText.c_str());
		*/
	}

	return(nPrompt);
}

int
CLLMContext::Sample(VChatMessagesA & pMessagesA)
{
	VChatMessages	pMessages;

	for (const auto & pMessageA : pMessagesA) {
		TChatMessage	pMessage;

		std::u8string	pOutText;
		::SJIStoUTF8(pMessageA.pRole.size(), pMessageA.pRole.c_str(), pOutText);
		size_t	nUTF8Text = pOutText.length();
		pMessage.pRole.resize(nUTF8Text + 1);
		memcpy(pMessage.pRole.data(), pOutText.data(), nUTF8Text);

		::SJIStoUTF8(pMessageA.pContent.size(), pMessageA.pContent.c_str(), pOutText);
		nUTF8Text = pOutText.length();
		pMessage.pContent.resize(nUTF8Text + 1);
		memcpy(pMessage.pContent.data(), pOutText.data(), nUTF8Text);

		pMessages.push_back(pMessage);
	}

	return(CLLMContext::Sample(pMessages));
}

int
CLLMContext::Sample(VChatMessagesW & pMessagesW)
{
	VChatMessages	pMessages;

	for (const auto & pMessageW : pMessagesW) {
		TChatMessage	pMessage;

		std::u8string	pOutText;
		::UCS2toUTF8(pMessageW.pRole.size(), pMessageW.pRole.c_str(), pOutText);
		size_t	nUTF8Text = pOutText.length();
		pMessage.pRole.resize(nUTF8Text + 1);
		memcpy(pMessage.pRole.data(), pOutText.data(), nUTF8Text);

		::UCS2toUTF8(pMessageW.pContent.size(), pMessageW.pContent.c_str(), pOutText);
		nUTF8Text = pOutText.length();
		pMessage.pContent.resize(nUTF8Text + 1);
		memcpy(pMessage.pContent.data(), pOutText.data(), nUTF8Text);

		pMessages.push_back(pMessage);
	}

	return(CLLMContext::Sample(pMessages));
}


int
CLLMContext::Sample(VChatMessages & pMessages)
{
	std::string		pPromptText;
	CChatTemplate::Apply(m_pLLM, pMessages, pPromptText);

	const char *	pPrompt = pPromptText.data();
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

	// 7. テキスト生成ループ
	int max_tokens = 64000;
	int i;
	for (i = 0; i < max_tokens; ++i) {
		// 次のトークンをサンプリング
		llama_token new_token = llama_sampler_sample(pSampler, m_pContext, -1);
		llama_sampler_accept(pSampler, new_token);

		// 終了トークン (EOS) か判定
		if (llama_vocab_is_eog(vocab, new_token)) {
			break;
		}

		// トークンを文字列に変換して出力
		char	buf[128];
		auto n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
		//auto n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
		if (n > 0) {
			if (m_pListener) {
				m_pListener->OnToken(n, buf);
			}
		}

		// 生成されたトークンをコンテキストに入力して次を予測
		batch = llama_batch_get_one(&new_token, 1);
		if (llama_decode(m_pContext, batch) != 0) {
			break;
		}
	}

	return(0);
}

