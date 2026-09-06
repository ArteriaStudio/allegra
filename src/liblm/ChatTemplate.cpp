// LLM操作（liblm）
#include	"pch.h"
#include	<misc/libtx/txText.h>
#include	<libux/UxTools.h>
#include	"liblm/liblm.h"



CChatTemplate::CChatTemplate()
{
}

CChatTemplate::~CChatTemplate()
{
}

#ifdef ENABLE_ASCII_PARAMETER
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
#endif

int32_t
CChatTemplate::ApplyPhi3Format(VChatMessages & pMessages, std::u8string & pPrompt, bool add_generation_prompt)
{
	for (const auto & msg : pMessages) {
		auto role_tag = msg.pRole;

		if (role_tag.compare(u8"system") == 0) {
			pPrompt += u8"<|system|>\n";
			pPrompt += msg.pContent;
			pPrompt += u8"<|end|>\n";
		} else if (role_tag.compare(u8"user") == 0) {
			pPrompt += u8"<|user|>\n";
			pPrompt += msg.pContent;
			pPrompt += u8"<|end|>\n";
		} else if (role_tag.compare(u8"assistant") == 0) {
			pPrompt += u8"<|assistant|>\n";
			pPrompt += msg.pContent;
			pPrompt += u8"<|end|>\n";
		}
	}
	
	if (add_generation_prompt) {
		pPrompt += u8"<|assistant|>\n";
	}

	return((int32_t)pPrompt.length());
}

int32_t
CChatTemplate::ApplyFormat(VChatMessages & pMessages, std::u8string & pPrompt, bool add_generation_prompt)
{
	for (const auto & msg : pMessages) {
		auto role_tag = msg.pRole;

		pPrompt += u8"<|im_start|>" + role_tag + u8"<|im_sep|>\n";
		pPrompt += msg.pContent;
		pPrompt += u8"<|im_end|>\n";
	}

	
	if (add_generation_prompt) {
		pPrompt += u8"<|im_start|>assistant<|im_sep|>\n";
	}
	

	return((int32_t)pPrompt.length());
}

//　Generate by Gemini3.7 Flash
//　Refactoring manual.
int32_t
CChatTemplate::ApplyGemmaFormat(VChatMessages & pMessages, std::u8string & pPrompt, bool add_generation_prompt)
{
	for (const auto & msg : pMessages) {
		auto role_tag = msg.pRole;
		// assistant を model に正規化
		if (role_tag == u8"assistant") {
			role_tag = u8"model";
		}

		pPrompt += u8"<|turn>" + role_tag + u8"\n";
		pPrompt += msg.pContent;
		pPrompt += u8"<turn|>\n";
	}

	if (add_generation_prompt) {
		pPrompt += u8"<|turn>model\n";
	}

	return((int32_t)pPrompt.length());
}

#ifdef ENABLE_ASCII_PARAMETER
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
#endif

//　
int32_t
CChatTemplate::Apply(CCtrlLLM * pLLM, VChatMessages & pMessages, std::u8string & pPrompt, bool add_generation_prompt)
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
	} else if (pArch.compare("phi3") == 0) {
		nPrompt = ApplyPhi3Format(pMessages, pPrompt, add_generation_prompt);
	} else {
		nPrompt = ApplyFormat(pMessages, pPrompt, add_generation_prompt);
	}

	return(nPrompt);
}
