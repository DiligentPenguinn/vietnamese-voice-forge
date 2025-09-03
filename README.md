# vietnamese-voice-forge
## 📖 Overview
This project focuses on **Vietnamese style paraphrasing** as an entry point into natural language processing (NLP). 
Specifically, we excecute two tasks: paraphrasing modern Vietnamese sentence to ancient style (cổ trang) Vietnamese sentence, and paraphrasing modern Vietnamese sentence to Quang Binh Dialect. 

*Note: the API is currently stopped, hence the UI will return error if you try to use. You can use the model that we upload and run it yourself*

Below are two examples of paraphrasing: 
- **Modern sentence**: "Anh ấy luôn cố gắng hết sức để bảo vệ gia đình mình." ->  **Ancient Vietnamese Sentence**: "Chàng dốc hết tâm can, nguyện lấy thân này che chở gia quyến."
- **Modern Sentence**: "Xin chào bạn, vì sao bạn xinh đẹp đến vậy" -> **QB Dialect**: "Chào mi, răng mi đẹp cấy rứa hè"
We compare two main approaches:

1. **Cost-efficient model fine-tuning**
2. **RAG pipelines leveraging DeepSeek V3**

In total, we trained and deployed 4 models. 2 finetuned models and 2 others model using RAG pipelines, Details will be specified below.

DeepSeek V3 was chosen for its **robustness in Vietnamese** and **cost-effectiveness** compared to other popular models such as OpenAI GPT-4o and Gemini.  

👉 The entire project was conducted under a budget of **$30**, showcasing the viability of low-cost NLP research for low-resource languages like Vietnamese.

--- 

## ⚙️ Methodology

### 1. Training & Data Collection
- **Fine-tuning models**:  
  - `Deepseek-r1-distill-qwen-1.5b-bnb-4bit`  
  - `DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit`  
  - Chosen for cost efficiency and effective fine-tuning capability.

 - **Fine-tuning methods**: Using Hugging Face Transformer library 

- **Dialect adaptation (Quảng Bình)**:  
  - Used DeepSeek V3 with a **rule file** to generate ~20,000 sentence pairs.  
  - We also tried OpenAI GPT-40 and found that DeepSeek V3 outperformed OpenAI GPT-4o in idiomatic accuracy (human-eval).

- **Classical style (Cổ Trang)**:  
  - Collected ~40,000 sentences ([Dataset link](https://huggingface.co/datasets/triettheeducator/modern-to-ancient-vietnamese-paraphrased-dataset))
 from classical novels:  
    *Hồng Lâu Mộng, Tam Quốc Diễn Nghĩa, Liêu Trai Chí Dị, Thủy Hử, Tây Du Ký*.  
  - Translated them into modern Vietnamese using GPT-4o Mini.  
  - Dataset is available on Hugging Face.

- **RAG pipeline**:  
  - Built using `RecursiveCharacterTextSplitter` from LangChain.  
  - Paired with DeepSeek V3 for semantic retrieval and paraphrasing.

---

### 2. Evaluation
- **Metrics**:  
  - **BERTScore** → semantic similarity. F1 score to be more specific
  - **Self-BLEU** → diversity of paraphrasing (coming soon) 

- **Test Dataset**:  
  - 1,000 Vietnamese sentences from [this dataset](https://huggingface.co/datasets/DiligentPenguinn/vietnamese-paraphrase-pairs-dataset)

- **Result**
+ QB Dialect finetuned: 0.669
+ Ancient Vietnamese fine-tuned: 0.773
+ Deepseek + Rag for QB Dialect: 0.914
+ Deepseek + Rag for Vietnamese ancient style paraphrasing: 0.551 (It seems that we need to update embedding model, so Bertscore can capture semantic similarity more correctly, as by human-evaluation, we see the paraphrased sentence is quite close to the original sentence, semantically. 

![Thiết kế chưa có tên](https://github.com/user-attachments/assets/58a71f5b-856e-4950-8ec2-fc94b7413f1b)


---

## 🚀 Deployment
- **Model hosting**: [VastAI](https://vast.ai/) (low-cost GPU inference)  
- **Backend**: [Supabase](https://supabase.com/)  
- **Frontend**: [Lovable](https://lovable.dev/)  

---

## Future Work

- Execute more NLP methods to increase paraphrasing quality
- Explore the dynamic impact of different languages such as Vietnamese and English to performance of small-size model (< 1B parameters)

## Acknowledgment
THis is our first project exploring NLP. There are so many rooms for improvement and many more things to learn. This is not by any mean the standard way for executing paraphrasing task. 

## 👥 Authors

- **Minh Triet Nguyen**  
  Graduate, Marketing Management – British University Vietnam  
  📧 Email: [Triet2one@gmail.com](mailto:Triet2one@gmail.com)

- **Duy Dang Hoang**  
  Junior, Computer Science – National University of Singapore  
  📧 Email: [hoangduyqb2209@gmail.com](mailto:hoangduyqb2209@gmail.com)

---
