"""Chat-UniVi LLaMA wrapper for ALS3P grounding tokens."""

from ChatUniVi.model.arch_ALS3P import (
    ChatUniViALS3PMetaForCausalLM,
)
from ChatUniVi.model.language_model.llama import ChatUniViLlamaForCausalLM


class ChatUniViALS3PLlamaForCausalLM(ChatUniViLlamaForCausalLM):
    def prepare_inputs_labels_for_multimodal(self, *args, **kwargs):
        return ChatUniViALS3PMetaForCausalLM.prepare_inputs_labels_for_multimodal(
            self, *args, **kwargs
        )
