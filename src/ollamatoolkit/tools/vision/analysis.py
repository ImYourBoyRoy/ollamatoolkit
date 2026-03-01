# ./ollamatoolkit/tools/vision/analysis.py
"""
Ollama Toolkit - Vision Analysis
================================
VQA and Code Generation logic.
"""


class AnalysisProcessor:
    def __init__(self, client):
        self.client = client

    def image_to_code(
        self, image_path: str, target_format: str = "HTML/Tailwind"
    ) -> str:
        """
        Converts a UI screenshot or mockup into code.
        """
        # Modern prompt optimized for VLM (Qwen/Llava)
        prompt = (
            f"Convert this UI image into detailed {target_format} code. "
            "Respond ONLY with the code block. "
            "Do not explain. Do not wrap in markdown quotes if possible, just the raw code or a single code block."
        )
        return self.client.analyze_image(image_path, prompt)

    def general_qa(self, image_path: str, question: str) -> str:
        """
        General Q&A on the image.
        """
        return self.client.analyze_image(image_path, question)
