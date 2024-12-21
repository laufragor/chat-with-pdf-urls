# Chat with online PDF documents

Load one or multiple PDFs from the web and use Google Gemini to generate answers to your questions.

## Installation

1. Clone the repository (or download the ZIP file and extract its contents):
   ```
   git clone https://github.com/laufragor/chat-with-pdf-urls
   cd chat-with-pdf-urls
   ```
   

2. Install the required dependencies:
    ```
    pip3 install -r requirements.txt
    ```

3. Add your Google GenAI key to the `.env` file as shown below:

    ```
    API_KEY=your_api_key_here
    ```

## Usage

### Running the script

    ```
    streamlit run app.py
    ```

### Asking questions

Once the script is running, you can ask questions about the PDF document. Type a question and press Enter. 
