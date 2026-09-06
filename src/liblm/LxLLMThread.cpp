// LLM操作（liblm）
#include	"pch.h"
#include	<misc/libtx/txText.h>
#include	<libux/UxTools.h>
#include	"liblm/liblm.h"



CLxLLMThread::CLxLLMThread() : CAxThread(L"CLxLLMThread")
{
	m_pListener = nullptr;
}

CLxLLMThread::~CLxLLMThread()
{
	m_pListener = nullptr;
}

//　モデルウェイトをロード
int
CLxLLMThread::Create(const char * pFilepath, ILLMListener * pListener)
{
	if (m_pEventEnd.Create(TRUE, FALSE) == false) {
		return(-1);
	}
	if (m_pShutdown.Create(TRUE, FALSE) == false) {
		return(-1);
	}

	m_pModelFilepath = pFilepath;
	m_pListener = pListener;

	if (CAxThread::Create() == false) {
		return(-1);
	}
	CAxThread::Resume();

	return(0);
}

//　
void
CLxLLMThread::Delete(void)
{
	m_pModelFilepath.clear();
	m_pListener = nullptr;
	m_pEventEnd.Delete();
	m_pShutdown.Delete();

	return;
}

//　
int
CLxLLMThread::WaitForEndWorker()
{
	return(m_pEventEnd.Wait());
}

int
CLxLLMThread::Shutdown()
{
	m_pShutdown.Set();

	return(0);
}

//　
void
CLxLLMThread::OnBegin()
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
CLxLLMThread::Looper(void)
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
CLxLLMThread::Main(void)
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
CLxLLMThread::OnResponse(const int32_t nText, const char8_t * pText)
{
	if (m_pListener) {
		m_pListener->OnResponse(nText, pText);
	}

	return;
}

//　
int
CLxLLMThread::OnProgress(float fValue)
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
