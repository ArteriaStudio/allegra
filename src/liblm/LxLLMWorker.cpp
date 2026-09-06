// LLM操作（liblm）
#include	"pch.h"
#include	<misc/libtx/txText.h>
#include	<libux/UxTools.h>
#include	<liblm/liblm.h>

//　定数宣言


CLxLLMWorker::CLxLLMWorker() : CAxHandler()
{
	m_pListener = nullptr;
}

CLxLLMWorker::~CLxLLMWorker()
{
	m_pListener = nullptr;
}

//　
uint32_t
CLxLLMWorker::Initialize(void)
{
	auto bResult = CAxHandler::Create(nullptr);
	if (bResult == false) {
		return(-1);
	}

	return(0);
}

//　
void
CLxLLMWorker::Finalize(void)
{
	m_pLLM.Delete();



	CAxHandler::Delete();

	return;
}

//　モデルウェイトをロード
int
CLxLLMWorker::CreateModel(const char * pFilepath, ILLMListener * pListener)
{
	if (m_pEventEnd.Create(TRUE, FALSE) == false) {
		return(-1);
	}
	if (m_pShutdown.Create(TRUE, FALSE) == false) {
		return(-1);
	}
	if (m_pLLM.Create(pFilepath, this)) {
		return(-1);
	}




	m_pModelFilepath = pFilepath;
	m_pListener = pListener;

	return(0);
}

//　
void
CLxLLMWorker::DeleteModel()
{
	m_pLLM.Delete();


	m_pModelFilepath.clear();
	m_pListener = nullptr;

	m_pEventEnd.Delete();
	m_pShutdown.Delete();

	return;
}

//　
int
CLxLLMWorker::WaitForEndWorker()
{
	return(m_pEventEnd.Wait());
}

int
CLxLLMWorker::Shutdown()
{
	m_pShutdown.Set();

	return(0);
}

//　
void
CLxLLMWorker::OnBegin()
{
	VChatMessages	pMessages;

	TChatMessage	pMessage;

	pMessage.pRole = u8"user";
	pMessage.pContent = u8"こんにちは。自己紹介をお願いできますか。";
	pMessages.push_back(pMessage);

	pMessage.pRole = u8"system";
	pMessage.pContent = u8"プレーンテキスト形式で出力してください。";
	pMessages.push_back(pMessage);

	u8stringstream	pStream;
	m_pContext.Sample(pMessages, pStream);

	return;
}

//　
uint32_t
CLxLLMWorker::Looper(void)
{
	auto iResult = m_pContext.CreateContext(m_pLLM, this);
	if (iResult) {
		return(iResult);
	}

	OnBegin();

	HANDLE	pHandles[1] = {};
	DWORD	nHandles = _countof(pHandles);

	pHandles[0] = m_pShutdown.GetHandle();
//	pHandles[1] = m_pRender.GetEventHandle();
//	pHandles[2] = m_pQueueAct.GetHandle();

	do {
		auto dwResult = ::WaitForMultipleObjects(nHandles, pHandles, FALSE, INFINITE);
		if (dwResult == WAIT_FAILED) {
			break;
		}
		if (dwResult == WAIT_OBJECT_0) {
			//　スレッド終了イベント
			break;
		}
		if (dwResult == WAIT_OBJECT_1) {
			//　再生キューへ充当する空きが発生
			//　キューとバッファーから再生データを獲得
		}
		if (dwResult == WAIT_OBJECT_2) {
			//　キュー充当イベント
			//　特別な処理はない。
			::OutputDebugString(L"empty queue\n");
		}
	} while (1);

	m_pContext.DeleteContext();

	return(0);
}

//　
uint32_t
CLxLLMWorker::Main(void)
{
	auto status = m_pLLM.Create(m_pModelFilepath.c_str(), this);
	if (status == 0) {
		status = Looper();
	}
	m_pLLM.Delete();
	m_pEventEnd.Set();

	return(status);
}

//　
void
CLxLLMWorker::OnResponse(const int32_t nText, const char8_t * pText)
{
	if (m_pListener) {
		m_pListener->OnResponse(nText, pText);
	}

	return;
}

//　
int
CLxLLMWorker::OnProgress(float fValue)
{
	auto status = m_pShutdown.Wait(0UL);
	if (status == WAIT_OBJECT_0) {
		return(WAIT_TIMEOUT);
	}
	if (m_pListener) {
		status = m_pListener->OnProgress(fValue);
	} else {
		status = 0;
	}
	return(status);
}
