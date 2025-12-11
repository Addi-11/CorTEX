import json
import re
import sys
import torch
from typing import Dict, List, Any
from dataclasses import dataclass

# Add LLaVA-Med to path
sys.path.insert(0, '/mnt/workspace/LLaVA-Med')

try:
    from llava.model import LlavaMistralForCausalLM
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Warning: Could not import LLaVA-Med specific classes, using standard transformers")
    AutoModelForCausalLM = AutoModelForCausalLM
    LlavaMistralForCausalLM = AutoModelForCausalLM

# Mock ToolUniverse SDK
class ToolUniverseSDK:
    """Mock implementation of ToolUniverse SDK"""
    
    @staticmethod
    def call_tool(tool_name: str, params: Dict[str, Any]) -> str:
        """Call tool from ToolUniverse"""
        mock_responses = {
            "FDA_get_mechanism_of_action_by_drug_name": 
                "Cisplatin works by binding to DNA and inhibiting its synthesis, causing apoptosis in cancer cells.",
            "FDA_get_indications_by_drug_name": 
                "Cisplatin is FDA approved for: Testicular cancer, ovarian cancer, bladder cancer, lung cancer, head and neck cancer.",
            "FDA_get_adverse_reactions_by_drug_name": 
                "Common adverse reactions: Nausea, vomiting, nephrotoxicity (kidney damage), ototoxicity (hearing loss), peripheral neuropathy, myelosuppression.",
            "FDA_get_drug_interactions_by_drug_name": 
                "Cisplatin interactions: Aminoglycosides (increased nephrotoxicity), Loop diuretics (enhanced ototoxicity), Bleomycin (increased pulmonary toxicity).",
            "FDA_get_contraindications_by_drug_name": 
                "Cisplatin contraindications: Severe renal impairment (creatinine clearance <30 mL/min), pre-existing hearing loss, pregnancy.",
            "FDA_get_drug_names_by_indication": 
                "Drugs for bladder cancer: Cisplatin, Gemcitabine, BCG vaccine (intravesical), Doxorubicin, Mitomycin C.",
        }
        
        key = f"{tool_name}".lower()
        for k, v in mock_responses.items():
            if k.lower() in key or key in k.lower():
                return v
        
        return f"Tool {tool_name} executed with parameters {params}"


@dataclass
class ToolCall:
    name: str
    params: Dict[str, Any]


class ToolCallParser:
    """Parse tool calls and final answers from model output"""
    
    # Known valid tools
    VALID_TOOLS = {
        "FDA_get_mechanism_of_action_by_drug_name",
        "FDA_get_indications_by_drug_name",
        "FDA_get_drug_interactions_by_drug_name",
        "FDA_get_adverse_reactions_by_drug_name",
        "FDA_get_contraindications_by_drug_name",
        "FDA_get_drug_names_by_indication",
        "FDA_get_boxed_warning_info_by_drug_name",
        "FDA_get_pregnancy_or_breastfeeding_info_by_drug_name",
        "OpenTargets_get_disease_ids_by_name",
        "OpenTargets_get_associated_drugs_by_disease_efoId",
        "DiseaseAnalyzerAgent",
        "TRIP_Database_Guidelines_Search",
        "PubMed_search_articles",
        "FAERS_search_adverse_event_reports",
    }
    
    @staticmethod
    def parse_params_flexible(params_str: str) -> Dict[str, Any]:
        """Parse parameters with multiple fallback strategies"""
        if not params_str:
            return {}
        
        params_str = params_str.strip()
        
        # Strategy 1: Direct JSON parse (with quote normalization)
        try:
            # Replace single quotes with double quotes
            normalized = params_str.replace("'", '"')
            return json.loads(normalized)
        except:
            pass
        
        # Strategy 2: Extract JSON object with regex
        try:
            match = re.search(r'\{[^{}]*\}', params_str)
            if match:
                json_str = match.group().replace("'", '"')
                return json.loads(json_str)
        except:
            pass
        
        # Strategy 3: Parse key=value or key: value patterns
        try:
            params = {}
            # Match patterns like drug_name="cisplatin" or drug_name: cisplatin
            kv_pattern = r'(\w+)\s*[=:]\s*["\']?([^"\',}\s]+)["\']?'
            matches = re.findall(kv_pattern, params_str)
            for key, value in matches:
                params[key] = value
            if params:
                return params
        except:
            pass
        
        # Strategy 4: Extract drug name heuristically
        try:
            drug_patterns = [
                r'drug[_\s]?name["\s:=]+["\']?(\w+)["\']?',
                r'"drug_name":\s*"([^"]+)"',
                r"'drug_name':\s*'([^']+)'",
            ]
            for pattern in drug_patterns:
                match = re.search(pattern, params_str, re.IGNORECASE)
                if match:
                    return {"drug_name": match.group(1).lower()}
        except:
            pass
        
        # Strategy 5: If just a word, assume it's the drug name
        if re.match(r'^[\w\s]+$', params_str) and len(params_str) < 50:
            return {"drug_name": params_str.strip().lower()}
        
        return {}
    
    @staticmethod
    def extract_tool_calls(output: str) -> List[ToolCall]:
        """Extract tool calls from model output - simple line-by-line parsing"""
        tool_calls = []
        
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or '---' in line or 'FINAL ANSWER' in line.upper():
                continue
            
            if ':' not in line:
                continue
            
            # Split on first colon
            parts = line.split(':', 1)
            tool_name = parts[0].strip()
            params_str = parts[1].strip() if len(parts) > 1 else ""
            
            # Check if valid tool name
            if tool_name not in ToolCallParser.VALID_TOOLS:
                continue
            
            # Parse params
            params = ToolCallParser.parse_params_flexible(params_str)
            if params:
                tool_calls.append(ToolCall(name=tool_name, params=params))
        
        return tool_calls
    
    @staticmethod
    def extract_final_answer(output: str) -> str:
        """Extract final answer from model output"""
        if "FINAL ANSWER:" in output:
            lines = output.split("FINAL ANSWER:")
            if len(lines) > 1:
                final_answer = lines[1].strip()
                # Remove the closing --- if present
                final_answer = final_answer.replace("---", "").strip()
                return final_answer
        return None

class LLaVAMedEvaluator:
    """End-to-end evaluator for LLaVA-Med with single tool call"""
    
    MODEL_PATH = "/mnt/workspace/CorTEX/.models/llava-med-v1.5-mistral-7b"
    
    TOOL_PROMPT = """You are a medical AI assistant. To answer the question, first call ONE tool.

TOOL FORMAT:
ToolName: {"param": "value"}

AVAILABLE TOOLS:
- FDA_get_mechanism_of_action_by_drug_name: {"drug_name": "X"} - HOW a drug works
- FDA_get_indications_by_drug_name: {"drug_name": "X"} - WHAT a drug treats
- FDA_get_adverse_reactions_by_drug_name: {"drug_name": "X"} - SIDE EFFECTS
- FDA_get_drug_interactions_by_drug_name: {"drug_name": "X"} - drug INTERACTIONS
- FDA_get_contraindications_by_drug_name: {"drug_name": "X"} - when NOT to use
- FDA_get_drug_names_by_indication: {"indication": "X", "limit": 5} - find drugs for a condition

Question: """

    ANSWER_PROMPT = """Based on the following information, provide a direct answer to the question.

Question: {question}

Information:
{tool_result}

Answer:"""

    def __init__(self, device: str = "cuda"):
        self.sdk = ToolUniverseSDK()
        self.parser = ToolCallParser()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"Loading LLaVA-Med model from {self.MODEL_PATH}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_PATH,
            trust_remote_code=True,
            use_fast=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Model loaded successfully")
    
    def _generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate response from model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def run_evaluation(self, question: str, expected_answer: str = None) -> Dict[str, Any]:
        """Run evaluation: one tool call, then answer"""
        
        result = {
            "question": question,
            "expected_answer": expected_answer,
            "tool_call": None,
            "tool_result": None,
            "final_answer": None,
            "evaluation": None
        }
        
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")
        
        # Step 1: Get tool call
        tool_prompt = self.TOOL_PROMPT + question
        tool_response = self._generate(tool_prompt, max_new_tokens=100)
        print(f"\nTool Response: {tool_response}")
        
        tool_calls = self.parser.extract_tool_calls(tool_response)
        
        if tool_calls:
            tool_call = tool_calls[0]  # Take first tool call only
            result["tool_call"] = {"name": tool_call.name, "params": tool_call.params}
            print(f"Tool: {tool_call.name} | Params: {tool_call.params}")
            
            # Execute tool
            tool_result = self.sdk.call_tool(tool_call.name, tool_call.params)
            result["tool_result"] = tool_result
            print(f"Result: {tool_result}")
            
            # Step 2: Get final answer (no tools in prompt)
            answer_prompt = self.ANSWER_PROMPT.format(question=question, tool_result=tool_result)
            final_answer = self._generate(answer_prompt, max_new_tokens=200)
        else:
            print("No tool call found, using direct response")
            final_answer = tool_response
        
        result["final_answer"] = final_answer
        print(f"\nFinal Answer: {final_answer}")
        
        # Evaluate
        if expected_answer:
            eval_result = self._evaluate(final_answer, expected_answer)
            result["evaluation"] = eval_result
            print(f"Match: {'✓' if eval_result['match'] else '✗'} ({eval_result['score']:.2f})")
        
        return result
    
    def _evaluate(self, answer: str, expected: str) -> Dict[str, Any]:
        """Simple keyword-based evaluation"""
        answer_lower = answer.lower()
        keywords = [w for w in expected.lower().split() if len(w) > 3]
        matches = sum(1 for kw in keywords if kw in answer_lower)
        score = matches / len(keywords) if keywords else 0
        return {"match": score > 0.4, "score": score}


def main():
    evaluator = LLaVAMedEvaluator()
    
    test_cases = [
        {
            "question": "What is the mechanism of action for cisplatin?",
            "expected_answer": "Cisplatin binds to DNA and inhibits synthesis"
        },
        {
            "question": "What are the side effects of cisplatin?",
            "expected_answer": "nephrotoxicity, ototoxicity, nausea, neuropathy"
        },
        {
            "question": "What drugs treat bladder cancer?",
            "expected_answer": "Cisplatin, Gemcitabine, BCG"
        },
    ]
    
    results = []
    for tc in test_cases:
        result = evaluator.run_evaluation(tc["question"], tc["expected_answer"])
        results.append(result)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        status = "✓" if r["evaluation"] and r["evaluation"]["match"] else "✗"
        print(f"{i}. {status} {r['question'][:50]}...")


if __name__ == "__main__":
    main()