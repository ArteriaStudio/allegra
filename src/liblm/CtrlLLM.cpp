// LLM操作（liblm）
#include	"pch.h"
#include	<misc/libtx/txText.h>
#include	<libux/UxTools.h>
#include	"liblm/liblm.h"


//　関数プロトタイプ
static bool 	model_load_progress_callback(float fProgress, void * user_data);

CCtrlLLM::CCtrlLLM()
{
	m_pModel = nullptr;
	m_pListener = nullptr;
}

CCtrlLLM::~CCtrlLLM()
{
	m_pListener = nullptr;
	m_pModel = nullptr;
}

//　
int
CCtrlLLM::Create(const char * pModelFilepath, ILLMListener * pListener)
{
	//　モデルをロード
	llama_model_params	model_params = llama_model_default_params();
	model_params.progress_callback = model_load_progress_callback;
	model_params.progress_callback_user_data = pListener;
//	model_params.progress_callback_user_data = this;
//	model_params.n_gpu_layers = 0; // GPUにオフロードするレイヤー数 (0でCPUのみ)
	model_params.n_gpu_layers = 99; // GPUにオフロードするレイヤー数 (0でCPUのみ)
	model_params.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
	model_params.check_tensors = false;

	m_pModel = ::llama_model_load_from_file(pModelFilepath, model_params);
	if (!m_pModel) {
		return((int)ECtrlLM::FailedLoadModel);
	}

	return(0);
}

//　
void
CCtrlLLM::Delete()
{
	if (m_pModel) {
		::llama_model_free(m_pModel);
		m_pModel = nullptr;
	}

	return;
}

llama_model *
CCtrlLLM::GetInterface(void)
{
	assert(m_pModel);
	return(m_pModel);
}

//　モデルロード進捗獲得
bool
model_load_progress_callback(float fProgress, void * user_data)
{
	auto pListener = (ILLMListener *)user_data;
	if (pListener) {
		auto status = pListener->OnProgress(fProgress);
		if (status) {
			// false を返すとロード処理を中断（キャンセル）できます
			return(false);
		}
	}

	return true; 
}
