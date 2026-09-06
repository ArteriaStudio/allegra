// LLM操作（liblm）
#pragma 	once
#include	<algorithm>
#include	<sstream>
#include	<libax/AxThread.h>
#include	<libax/AxHandler.h>
#include	<libax/AxEvent.h>
#include	<llama.h>


//　定数宣言
enum class	ECtrlLM : int
{
	Success = 0,
	FailedLoadModel,
	FailedCreateModel,
};

using u8stringstream = std::basic_stringstream<char8_t>;

//　LLM 対話メッセージ
struct	TChatMessage {
	std::u8string	pRole;			//　役割（文字コード：UTF-8）
	std::u8string	pContent;		//　本文（文字コード：UTF-8）
};
typedef std::vector<TChatMessage>	VChatMessages;

struct	TChatMessageA {
	std::string 	pRole;			//　役割（文字コード：ASCII/SJIS）
	std::string 	pContent;		//　本文（文字コード：ASCII/SJIS）
};
typedef std::vector<TChatMessageA>	VChatMessagesA;

struct	TChatMessageW {
	std::wstring	pRole;			//　役割（文字コード：UCS2）
	std::wstring	pContent;		//　本文（文字コード：UCS2）
};
typedef std::vector<TChatMessageW>	VChatMessagesW;

//　イベントリスナー
class	ILLMListener
{
public:
	virtual void	OnResponse(const int32_t nText, const char8_t * pText) = 0;
	virtual int 	OnProgress(float fProgress) = 0;
};


//　llama.cpp バックエンドラップ
class	CLlama
{
private:
	explicit CLlama();
	virtual ~CLlama();

protected:

public:
	int 	Initialize();
	void	Finalize();

	static CLlama * 	GetInstance(void);
};

//　LLMモデル
class	CCtrlLLM
{
protected:
	llama_model *	m_pModel;
	ILLMListener *	m_pListener;

public:
	explicit CCtrlLLM();
	virtual ~CCtrlLLM();

	int 	Create(const char * pModelFilepath, ILLMListener * pListener);
	void	Delete();

	llama_model *	GetInterface(void);
};

//　LLMモデルコンテキスト
class	CLLMContext
{
private:
	ILLMListener *	m_pListener;

protected:
	llama_context * 	m_pContext;
//	llama_sampler *		m_pGrammerSampler;
	std::string			m_pGrammar;
	CCtrlLLM *			m_pLLM;

	void	OnResponse(u8stringstream & pStream, const int32_t nText, const char8_t * pText);

public:
	explicit CLLMContext();
	virtual ~CLLMContext();

	int 	CreateContext(CCtrlLLM & pLLM, ILLMListener * pListener);
	void	DeleteContext();

	int 	Sample(VChatMessages & pMessages, u8stringstream & pStream);
	int 	Sample(VChatMessagesA & pMessages, u8stringstream & pStream);
	int 	Sample(VChatMessagesW & pMessages, u8stringstream & pStream);
};

//　チャット文言テンプレート
class	CChatTemplate
{
protected:
//	static int32_t	ApplyGemmaFormat(VChatMessages & pMessages, std::string & pPrompt, bool add_generation_prompt);
	static int32_t	ApplyGemmaFormat(VChatMessages & pMessages, std::u8string & pPrompt, bool add_generation_prompt);
	static int32_t	ApplyPhi3Format(VChatMessages & pMessages, std::u8string & pPrompt, bool add_generation_prompt);
	static int32_t	ApplyFormat(VChatMessages & pMessages, std::u8string & pPrompt, bool add_generation_prompt);

public:
	explicit CChatTemplate();
	virtual ~CChatTemplate();

//	static int32_t	Apply(CCtrlLLM * pLLM, VChatMessages & pMessages, std::string & pPrompt, bool add_generation_prompt=true);
	static int32_t	Apply(CCtrlLLM * pLLM, VChatMessages & pMessages, std::u8string & pPrompt, bool add_generation_prompt=true);
};

//　LLM スレッド
class	CLxLLMThread : private CAxThread, public ILLMListener
{
private:
	CAxEvent	m_pShutdown;	//　スレッド停止イベント
	CAxEvent	m_pEventEnd;	//　スレッド停止完了イベント

	ILLMListener *	m_pListener;

protected:
	CCtrlLLM		m_pLLM;
	CLLMContext 	m_pContext;
	std::string 	m_pModelFilepath;

	uint32_t	Main(void);
	uint32_t	Looper(void);

	void	OnBegin();

public:
	explicit CLxLLMThread();
	virtual ~CLxLLMThread();

	int 	Create(const char * pFilepath, ILLMListener * pListener);
	void	Delete();

	int 	WaitForEndWorker();
	int 	Shutdown(void);

	void	OnResponse(const int32_t nText, const char8_t * pText);
	int 	OnProgress(float fValue);
};

//　LLM ワーカーハンドラ
class	CLxLLMWorker : public CAxHandler, private ILLMListener
{
private:
	CAxEvent	m_pShutdown;	//　スレッド停止イベント
	CAxEvent	m_pEventEnd;	//　スレッド停止完了イベント

	ILLMListener *	m_pListener;

protected:
	CCtrlLLM		m_pLLM;
	CLLMContext 	m_pContext;
	std::string 	m_pModelFilepath;

	uint32_t	Main(void);
	uint32_t	Looper(void);

	void	OnBegin();

public:
	explicit CLxLLMWorker();
	virtual ~CLxLLMWorker();

	uint32_t	Initialize();
	int 		CreateModel(const char * pFilepath, ILLMListener * pListener);
	void		DeleteModel();
	void		Finalize();


	int 	WaitForEndWorker();
	int 	Shutdown(void);

	void	OnResponse(const int32_t nText, const char8_t * pText);
	int 	OnProgress(float fValue);
};
