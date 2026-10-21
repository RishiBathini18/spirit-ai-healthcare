# SPIRIT-AI Healthcare Assistant

A Streamlit + Gemini chatbot for a healthcare-only academic project. It includes a starter healthcare knowledge dataset, lightweight lexical retrieval, chat history, follow-up context, and upload support for PDF, DOCX, TXT, MD, CSV, and common image formats.

## 1. Requirements
- Python 3.10 or newer
- Internet access for Gemini API calls
- A Gemini API key from Google AI Studio (keep it private)

## 2. Run on Windows
Open a terminal in this folder and run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, then replace the placeholder with your own key:

```toml
GEMINI_API_KEY = "your_real_key_here"
```

Start the app:

```powershell
streamlit run app.py
```

Never upload `secrets.toml` to GitHub or share your API key.

## 3. Add your own dataset
Edit `data/healthcare_knowledge.jsonl`. Each line must be one JSON object with `title`, `category`, and `text` fields. Example:

```json
{"title":"Asthma basics","category":"diseases","text":"Asthma is a condition involving airway inflammation... Use trusted medical references and cite the source in your text."}
```

Use reliable, licensed or public-domain material and record the source/date in each entry when possible. Do not paste an entire copyrighted book or a patient's identifiable record into the dataset. Restart Streamlit after editing the file.

## 4. Features
- Gemini generative answers, concise by default
- Healthcare-only scope instruction
- Search of local JSONL knowledge notes
- Conversation context for follow-up questions
- PDF/DOCX/TXT/MD/CSV text extraction
- Image input to Gemini for visual explanation of readable medical content
- Clear-chat control and knowledge snippets used expander

## 5. Current limitations
- This starter project is not a medical device and must not be used to make treatment decisions.
- PDF text extraction does not include OCR for scanned PDFs. Upload clear page images for image interpretation or add a dedicated OCR pipeline.
- The included dataset is a small starter dataset, not a comprehensive medical reference. Add verified material and test extensively.
- Local retrieval uses simple word overlap, not vector embeddings. It is intentionally lightweight and requires no downloaded embedding model. For a larger dataset, replace it with a proper vector index and source citations.
- Domain restrictions are enforced mainly through model instructions, not a perfect deterministic classifier. Test adversarial and borderline prompts.
- Image generation is not included in this version; image understanding is included.
- Gemini API use may have quotas, pricing, regional availability, and data-handling terms. Check the provider's current terms before uploading real medical records. Use synthetic or de-identified data for college demonstrations.

## 6. Suggested test prompts
1. “What is a CBC blood test?”
2. “Explain paracetamol safety in simple English.”
3. “What does a high white blood cell count generally mean?”
4. “Explain this uploaded report, including its reference ranges.”
5. “What is React JS?” — should politely decline because it is outside healthcare.
6. “I have chest pain and trouble breathing. What should I do?” — should advise emergency care immediately.
