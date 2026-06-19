"""
LLM Commentary Enhancer.

This module leverages a 4-bit quantized LLaMA-3 model to rewrite and enhance
basic algorithmic match events into rich, broadcast-quality sports commentary.
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class LLMCommentaryEnhancer:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        """
        Initializes the LLM and tokenizer using 4-bit quantization.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing LLM Enhancer with {model_name}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load with 4-bit quantization for VRAM efficiency (e.g. 4GB GPUs)
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            logger.info("Successfully loaded LLaMA-3 in 4-bit quantization mode!")
        except Exception as e:
            logger.error(f"Failed to load LLM ({e}). Ensure you have logged in to huggingface-cli.")
            raise

    def enhance_event(self, basic_event_text: str) -> str:
        """
        Rewrites a boring template string into an exciting broadcast commentary.
        """
        prompt = (
            "You are an energetic, professional football (soccer) television commentator. "
            "Rewrite the following basic match event into a single, exciting, and natural broadcast sentence. "
            "Do not add extra information that isn't implied. Do not include quotes or conversational filler. "
            f"Event: '{basic_event_text}'\n"
            "Commentary:"
        )

        try:
            # Tokenize prompt and ensure input is on the right device
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Because of auto device mapping, input_ids generally need to be on the primary device
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate the response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=40,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract only the newly generated tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            enhanced_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Clean up potential artifacts
            enhanced_text = enhanced_text.replace('"', '').replace('\n', ' ').strip()
            
            return enhanced_text
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return basic_event_text  # Fallback to original text if generation fails
