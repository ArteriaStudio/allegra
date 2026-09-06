// LLM操作（liblm）
#include	"pch.h"
#include	<misc/libtx/txText.h>
#include	<libux/UxTools.h>
#include	"liblm/liblm.h"

//　関数プロトタイプ
static void 	log_callback_null(ggml_log_level level, const char * text, void * user_data);

//　インポートライブラリ
#pragma comment(lib, "ggml.lib")
#pragma comment(lib, "ggml-base.lib")
#pragma comment(lib, "ggml-cpu.lib")
#pragma comment(lib, "ggml-vulkan.lib")
#pragma comment(lib, "D:/App/Vulkan/lib/vulkan-1.lib")
#pragma comment(lib, "llama.lib")
#pragma comment(lib, "llama-common.lib")






static void
log_callback_null(ggml_log_level level, const char* text, void* user_data) {
	(void)level;
	(void)text;
	(void)user_data;
}

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
//	::llama_log_set(log_callback_null, nullptr);
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

CLlama *
CLlama::GetInstance(void)
{
static CLlama	pInstance;

	return(&pInstance);
}
