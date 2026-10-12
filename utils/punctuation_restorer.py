from deepmultilingualpunctuation import PunctuationModel
import torch

# class PunctuationRestorer:
    # def __init__(self, model_name="oliverguhr/fullstop-punctuation-multilang-large"):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = PunctuationModel(model=model_name)
    
    # def restore(self, text):
        # if not text or not isinstance(text, str):
            # return text
            
        # try:
            # return self.model.restore_punctuation(text)
        # except Exception as e:
            # print(f"Error restoring punctuation: {e}")
            # return text
  

  
import nemo.collections.nlp.models as nemo_nlp

class PunctuationRestorer:
    def __init__(self):
        self.model = nemo_nlp.PunctuationCapitalizationModel.from_pretrained(
            # "punctuation_en_bert"
            "punctuation_en_distilbert"
        )
        print(f"PunctuationRestorer={nemo_nlp.PunctuationCapitalizationModel.list_available_models()}")
        
    def restore(self, text):
        if not text.strip():
            return text
            
        results = self.model.add_punctuation_capitalization([text])
        return results[0]