# vietnamese-voice-forge
## 📖 Overview
This project focuses on **Vietnamese paraphrasing** as an entry point into natural language processing (NLP).  
We compare two main approaches:
1. **Cost-efficient model fine-tuning**
2. **RAG pipelines leveraging DeepSeek V3** 

DeepSeek V3 was chosen for its **robustness in Vietnamese** and **cost-effectiveness** compared to other popular models such as OpenAI GPT-4o and Gemini.  

👉 The entire project was conducted under a budget of **$30**, showcasing the viability of low-cost NLP research for low-resource languages like Vietnamese.

--- 

## ⚙️ Methodology

### 1. Training & Data Collection
- **Fine-tuning models**:  
  - `deepseek-r1-distill-qwen-1.5b-bnb-4bit`  
  - `DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit`  
  - Chosen for cost efficiency and effective fine-tuning capability.

 - **Fine-tuning methods**: Using Hugging Face Transformer library 

- **Dialect adaptation (Quảng Bình)**:  
  - Used DeepSeek V3 with a **rule file** to generate ~20,000 sentence pairs.  
  - DeepSeek V3 outperformed OpenAI GPT-4o in idiomatic accuracy.

- **Classical style (Cổ Trang)**:  
  - Collected ~40,000 sentences from classical novels:  
    *Hồng Lâu Mộng, Tam Quốc Diễn Nghĩa, Liêu Trai Chí Dị, Thủy Hử, Tây Du Ký*.  
  - Translated them into modern Vietnamese using GPT-4o Mini.  
  - Dataset is available on Hugging Face.

- **RAG pipeline**:  
  - Built using `RecursiveCharacterTextSplitter` from LangChain.  
  - Paired with DeepSeek V3 for semantic retrieval and paraphrasing.

---

### 2. Evaluation
- **Metrics**:  
  - **BERTScore** → semantic similarity  
  - **Self-BLEU** → diversity of paraphrasing (coming soon) 

- **Test Dataset**:  
  - 1,000 Vietnamese sentences (source: to be specified).

---

## 🚀 Deployment
- **Model hosting**: [VastAI](https://vast.ai/) (low-cost GPU inference)  
- **Backend**: [Supabase](https://supabase.com/)  
- **Frontend**: [Lovable](https://lovable.dev/)  

---

## Future Work

- Execute more NLP methods to increase paraphrasing quality
- Explore the dynamic impact of different languages such as Vietnamese and English to performance of small-size model (< 1B parameters) 

## 👥 Authors

- **Minh Triet Nguyen**  
  Graduate, Marketing Management – British University Vietnam  
  📧 Email: [Triet2one@gmail.com](mailto:Triet2one@gmail.com)

- **Duy Dang Hoang**  
  Junior, Computer Science – National University of Singapore  
  📧 Email: [hoangduyqb2209@gmail.com](mailto:hoangduyqb2209@gmail.com)

---
