// LLM操作（liblm）
#pragma 	once
#include	<llama.h>

//　定数宣言
enum class	ECtrlLM : int
{
	Success = 0,
	FailedLoadModel,
	FailedCreateModel,
};

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
	virtual void	OnToken(const int32_t nText, const char * pText) = 0;
};


//　llama.cpp バックエンドラップ
class	CLlama
{
protected:

public:
	explicit CLlama();
	virtual ~CLlama();

	int 	Initialize();
	void	Finalize();
};

//　LLMモデル
class	CCtrlLLM
{
protected:
	llama_model *	m_pModel;

public:
	explicit CCtrlLLM();
	virtual ~CCtrlLLM();

	int 	Create();
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
	CCtrlLLM *	m_pLLM;

public:
	explicit CLLMContext();
	virtual ~CLLMContext();

	int 	CreateContext(CCtrlLLM & pLLM, ILLMListener * pListener);
	void	DeleteContext();

	int 	Sample(VChatMessages & pMessages);
	int 	Sample(VChatMessagesA & pMessages);
	int 	Sample(VChatMessagesW & pMessages);
};

//　チャット文言テンプレート
class	CChatTemplate
{
protected:
//	static int32_t	ApplyGemmaFormat(VChatMessages & pMessages, std::string & pPrompt, bool add_generation_prompt);
	static int32_t	ApplyGemmaFormat(VChatMessages & pMessages, std::u8string & pPrompt, bool add_generation_prompt);

public:
	explicit CChatTemplate();
	virtual ~CChatTemplate();

//	static int32_t	Apply(CCtrlLLM * pLLM, VChatMessages & pMessages, std::string & pPrompt, bool add_generation_prompt=true);
	static int32_t	Apply(CCtrlLLM * pLLM, VChatMessages & pMessages, std::u8string & pPrompt, bool add_generation_prompt=true);
};

