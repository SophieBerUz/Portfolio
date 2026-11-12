from __future__ import annotations

import os
import re
import json
from typing import Dict, List, Tuple
from itertools import product
from tqdm import tqdm
import openai
from openai import OpenAI
from dotenv import load_dotenv
import random

from locisimiles.document import Document, TextSegment

# ================= ENV =================

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# ================= TYPES =================

PairKey = Tuple[str,str] #(source_id, query_id)
FinalResults = Dict[PairKey, list[int]]
FullDict = Dict[str, List[Tuple[TextSegment, float, float]]] #Evaluator Format

# ================= Convert Results to FullDict =================

def convert_llm_results(final_results: FinalResults, source_doc, threshold: float = 0.5) -> FullDict:
    """
    Convert FinalResults from LLM pipeline to FullDict for IntertextEvaluator.

    - aggregate 0/1 labels per pair to probability
    - Set similarity to Dummy-value (1.0)
    - organize dictionary after query_id → List[(source_segment, sim, prob)]
    """
    out: FullDict = {}

    for (source_id, query_id), labels in final_results.items():
        prob = sum(labels) / len(labels)  # Durchschnitt over all windows
        if prob >= threshold:  # optional Threshold
            src_segment = source_doc[source_id]  # TextSegment from source
            out.setdefault(query_id, []).append((src_segment, 1.0, prob))

    return out

def convert_llm_results_coverage_fair(final_results: FinalResults,
    source_doc: Document,
    query_doc: Document,
    source_windows: list[list[TextSegment]],
    query_windows: list[list[TextSegment]],
    threshold: float = 0.5,
) -> FullDict:
    """
    Convert FinalResults to FullDict for IntertextEvaluator,
    with scores normalized according to segment coverage.

    - aggregate 0/1 labels per pair to probability
    - normalize scores by true window coverage for each segment 
    - set similarity to dummy value 1.0
    """
    out: FullDict = {}
    
    #calculate true coverage for each segment
    coverage_source = {s.id: 0 for s in source_doc.segments.values()}
    coverage_query  = {q.id: 0 for q in query_doc.segments.values()}
    
    for w in source_windows:
        for s in w:
            if not s.id.startswith("__PAD__"):
                coverage_source[s.id] += 1

    for w in query_windows:
        for q in w:
            if not q.id.startswith("__PAD__"):
                coverage_query[q.id] += 1
    
    # Build final FullDict
    for (source_id, query_id), labels in final_results.items():
        # Tatschliche Coverage pro Paar
        max_labels = coverage_source.get(source_id, 1) * coverage_query.get(query_id, 1)
        if max_labels == 0:
            print(f"[SKIP] max_labels=0 for {query_id} || {source_id}")
            continue
        sum_labels = sum(labels)
        len_labels = len(labels)
        prob = sum(labels) / max_labels
        
        # =========================
        # DEBUG PRINT
        # =========================
        #if sum_labels > 0:
            #print(f"\nPAIR: {query_id} || {source_id}")
            #print(f"  labels: {labels}")
            #print(f"  sum(labels): {sum_labels}")
            #print(f"  len(labels): {len_labels}")
            #print(f"  coverage_source: {coverage_source.get(source_id, 1)}")
            #print(f"  coverage_query: {coverage_query.get(query_id, 1)}")
            #print(f"  max_labels: {max_labels}")
            #print(f"  prob: {prob:.4f}")

            # optional: Verhältnis
            #observed_ratio = len_labels / max_labels if max_labels > 0 else 0
            #print(f"  observed/max: {observed_ratio:.4f}")

            #print("-" * 50)

        if prob >= threshold:
            src_segment = source_doc[source_id]
            out.setdefault(query_id, []).append((src_segment, 1.0, prob))

    return out
    
# ================== JSON-SAVE FORMAT ===================
def escape_ts(ts: TextSegment) -> tuple[str, str]:
    """
    Escape ID und Text of a  TextSegment, so that it is JSON-save.
    return Tuple: (escaped_id, escaped_text)
    """
    return json.dumps(ts.id), json.dumps(ts.text)
# ================== SAVE/LOAD RESULTS ===================
def save_final_results(
   final_results : FinalResults,
   path: str,
   *,
   meta : dict | None = None, 
):
    """
    Save FinalResults to disk so that the pipeline
    does NOT need to be rerun for evaluation.

    Format:
    {
      "meta": {...},
      "results": {
        "source_id|||query_id": [0,1,1,0,...],
        ...
      }
    }
    """
    payload = {
        "meta": meta or {},
        "results": {
            f"{source_id}|||{query_id}": labels
            for (source_id, query_id), labels in final_results.items()
        }
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        


def load_final_results(path: str) -> FinalResults:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    out: FinalResults = {}
    for k, v in payload["results"].items():
        source_id, query_id = k.split("|||")
        out[(source_id, query_id)] = v
    return out

def save_partial_results(
    final_results: FinalResults,
    path: str,
    source_window_idx: int,
) -> None:
    payload = {
        "source_window_idx": source_window_idx,
        "results": {
            f"{sid}|||{qid}": labels
            for (sid, qid), labels in final_results.items()
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_partial_results(path: str) -> tuple[int, FinalResults]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    final_results: FinalResults = {
        tuple(k.split("|||")): v
        for k, v in payload["results"].items()
    }

    return payload["source_window_idx"], final_results

# ================== PREDEFINED FEW-SHOT EXAMPLES ==================
# Format: List of tuples (query_window, source_window, results)
# results = [{"source_id": ..., "query_id": ..., "label": 1}, ...]
FEW_SHOT_EXAMPLES: list[tuple[list[TextSegment], list[TextSegment], list[dict]]] = [
    # Example 1
    (
        [TextSegment(text="O decus Italiae, uirgo!", seg_id="Q1")],
        [TextSegment(text="o decus Italiae uirgo, quas dicere grates quasue referre parem?",seg_id="S1",)],
        [{"source_id": "S1", "query_id": "Q1", "label": 1}]
    ),
    # Example 2
    (
        [TextSegment(text="Uirgilianum consilium est: Coniugium uocat, hoc praetexit nomine culpam.", seg_id="Q2")],
        [TextSegment(text="o decus Italiae uirgo, quas dicere grates quasue referre parem?", seg_id="S1")],
        [{"source_id": "S1", "query_id": "Q2", "label": 0}]
    ),
    # Example 3
    (
        [TextSegment(text="Hoc totum dico, quod non omnia possumus omnes: rarusque aut nullus est diuitum, qui in omni substantia sua paria uniuersa possideat.", seg_id="Q3")],
        [TextSegment(text="Haec Damon; uos, quae responderit Alphesiboeus, dicite, Pierides: non omnia possumus omnes.", seg_id="S2")],
        [{"source_id": "S2", "query_id": "Q3", "label": 1}]
    ),
    # Example 4
    (
        [TextSegment(text="Numquam hodie effugies, ueniam quocumque uocaris.", seg_id="Q4")],
        [TextSegment(text="Haec Damon; uos, quae responderit Alphesiboeus, dicite, Pierides: non omnia possumus omnes.", seg_id="S2")],
        [{"source_id": "S2", "query_id": "Q4", "label": 0}]
    ),
    # Example 5
    (
        [TextSegment(text="Quod si paululum se remiserit, quomodo qui aduerso flumine lembum trahit, si remiserit manus, statim retrolabitur, et fluentibus aquis, quo non uult ducitur  sic humana conditio, si paululum se remiserit, discit fragilitatem suam, et multa se non posse cognoscit.", seg_id="Q5")],
        [TextSegment(text="uidi lecta diu et multo spectata labore degenerare tamen, ni uis humana quotannis maxima quaeque manu legeret: sic omnia fatis in peius ruere ac retro sublapsa referri, non aliter quam qui aduerso uix flumine lembum remigiis subigit, si bracchia forte remisit, atque illum in praeceps prono rapit alueus amni.", seg_id="S3")],
        [{"source_id": "S3", "query_id": "Q5", "label": 1}]
    ),
    # Example 6
    (
        [TextSegment(text="Non tu in triuiis, indocte, solebas, stridenti miserum stipula disperdere carmen.", seg_id="Q6")],
        [TextSegment(text="uidi lecta diu et multo spectata labore degenerare tamen, ni uis humana quotannis maxima quaeque manu legeret: sic omnia fatis in peius ruere ac retro sublapsa referri, non aliter quam qui aduerso uix flumine lembum remigiis subigit, si bracchia forte remisit, atque illum in praeceps prono rapit alueus amni.", seg_id="S3")],
        [{"source_id": "S3", "query_id": "Q6", "label": 0}]
    ),
    # Example 7
    (
        [TextSegment(text="Sic pater ille Deum faciat: sic magnus Iesus.", seg_id="Q7")],
        [TextSegment(text="sic pater ille deum faciat, sic altus Apollo!", seg_id="S4")],
        [{"source_id": "S4", "query_id": "Q7", "label": 1}]
    ),
    # Example 8
    (
        [TextSegment(text="Obiiciunt mihi sectatores eius, cerealiaque arma expediunt fessi rerum, quare libros Origenis noxios et Ecclesiasticae fidei repugnantes in Latinum sermonem uerterim.", seg_id="Q8")],
        [TextSegment(text="sic pater ille deum faciat, sic altus Apollo!", seg_id="S4")],
        [{"source_id": "S4", "query_id": "Q8", "label": 0}]
    ),
    # Example 9
    (
        [TextSegment(text="Miror quomodo oblitus sis illos uersiculos ponere: unde tremor terris, qua ui maria alta tumescant, obicibus ruptis, rursusque in se ipsas residant defectus Solis uarios, Lunaeque laboresunde hominum genus et pecudum, unde imber et ignes, Arcturum, pluuiasque Hyadas, geminosque Triones.", seg_id="Q9")],
        [TextSegment(text="Me uero primum dulces ante omnia Musae, quarum sacra fero ingenti percussus amore, accipiant caelique uias et sidera monstrent, defectus solis uarios lunaeque labores, unde tremor terris, qua ui maria alta tumescant obicibus ruptis rursusque in se ipsa residant, quid tantum Oceano properent se tingere soles hiberni, uel quae tardis mora noctibus obstet.", seg_id="S5")],
        [{"source_id": "S5", "query_id": "Q9", "label": 1}]
    ),
    # Example 10
    (
        [TextSegment(text="Quod si in his uersiculis, quae de natura rerum scripsit, non est error, quomodo in aliis, quae de fidei dogmate scripsit, error esse potuit?", seg_id="Q10")],
        [TextSegment(text="Me uero primum dulces ante omnia Musae, quarum sacra fero ingenti percussus amore, accipiant caelique uias et sidera monstrent, defectus solis uarios lunaeque labores, unde tremor terris, qua ui maria alta tumescant obicibus ruptis rursusque in se ipsa residant, quid tantum Oceano properent se tingere soles hiberni, uel quae tardis mora noctibus obstet.", seg_id="S5")],
        [{"source_id": "S5", "query_id": "Q10", "label": 0}]
    ),
]
# ================== SLIDING WINDOW ==================
def fixed_sliding_windows(
    segments: List[TextSegment],
    k: int,
    stride: int=1
) -> List[List[TextSegment]]:
    """
    Generate fixed-size sliding windows.
    Window size is always exactly k.
    """
    if len(segments) < k:
        return []

    return [
        segments[i:i + k]
        for i in range(0,len(segments) - k + 1, stride)
    ]
# ================== PADDING ==================
def pad_segments(
    segments: List[TextSegment],
    k: int,
    prefix: str
) -> List[TextSegment]:
    """
    Add virtual padding segments at start and end so that
    each real segment appears in the same number of windows.
    """
    pad = k - 1
    padded: List[TextSegment] = []

    for i in range(pad):
        padded.append(
            TextSegment(
                text="(padding context; no textual content)",
                seg_id=f"__PAD__{prefix}_START_{i}"
            )
        )

    padded.extend(segments)

    for i in range(pad):
        padded.append(
            TextSegment(
                text="(padding context; no textual content)",
                seg_id=f"__PAD__{prefix}_END_{i}",
            )
        )

    return padded

# ================= JSON SCHEMA =================

INTEXTUALITY_SCHEMA = {
  "name": "intertextuality",
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "results": {
        "type": "array",
        "description": "ONLY intertextual (source_id, query_id) pairs in the window",
        "items": {
          "type": "object",
          "additionalProperties": False,
          "properties": {
            "source_id": {
              "type": "string",
              "description": "ID of the source segment"
            },
            "query_id": {
              "type": "string",
              "description": "ID of the query segment"
            },
            "label": {
              "type": "integer",
              "enum": [1],
              "description": "label 1 for intertextual"
            }
          },
          "required": ["source_id", "query_id", "label"]
        }
      }
    },
    "required": ["results"]
  }
}


# ================= PROMPT =================
def build_zero_shot_minimal_prompt(query_window: List[TextSegment], source_window: List[TextSegment]) -> str:
    query_block = "\n".join(f"[{escape_ts(q)[0]}] {escape_ts(q)[1]}" for q in query_window)
    source_block = "\n".join(f"[{escape_ts(s)[0]}] {escape_ts(s)[1]}" for s in source_window)
    return f"""
Given the following chunk from the query document:

{query_block}

And the following chunk from the source document:

{source_block}

Evaluate EACH possible (source_id, query_id) pair independently and 
determine whether the pair is an intertextuality.
Also use the surrounding chunk context to inform your judgment.

Return ONLY a valid JSON object in the specified schema.
""".strip()

def build_zero_shot_medium_prompt(query_window: List[TextSegment], source_window: List[TextSegment]) -> str:
    
    query_block = "\n".join(f"[{escape_ts(q)[0]}] {escape_ts(q)[1]}" for q in query_window)
    source_block = "\n".join(f"[{escape_ts(s)[0]}] {escape_ts(s)[1]}" for s in source_window)
    return f"""
You are a Latin philologist specializing in intertextuality. 
You are analyzing chunks of Latin text passages from different authors. 
Your goal is to identify which pairs exhibit intertextual relationships (e.g., quotation, allusion, paraphrase, or stylistic imitation). 
Consider similarities in wording, phrasing, imagery, or thematic correspondence. Ignore coincidental overlaps or generic expressions.

Given the following chunk from the query document:

{query_block}

And the following chunk from the source document:

{source_block}

Evaluate EACH possible (source_id, query_id) pair independently.
Also use the surrounding chunk context to inform your judgment.

Return ONLY a valid JSON object in the specified schema.
""".strip()

def build_zero_shot_maximum_prompt(query_window: List[TextSegment], source_window: List[TextSegment]) -> str:
    query_block = "\n".join(f"[{escape_ts(q)[0]}] {escape_ts(q)[1]}" for q in query_window)
    source_block = "\n".join(f"[{escape_ts(s)[0]}] {escape_ts(s)[1]}" for s in source_window)
    return f"""
You are a Latin philologist specializing in intertextuality. 
You are analyzing chunks of Latin text passages from different authors. 
Your goal is to identify which pairs exhibit intertextual relationships (e.g., quotation, allusion, paraphrase, or stylistic imitation). 
Consider similarities in wording, phrasing, imagery, or thematic correspondence. Ignore coincidental overlaps or generic expressions.

To achieve this, compare sentences on the following linguistic levels:
- morphological
- syntactic
- lexical
- idiomatic
- phraseological

Pay special attention to sentence pairs in which at least TWO of these
linguistic levels show meaningful agreement.
Note that NOT all linguistic characteristics need to be present
for a pair to be considered intertextual.

You MUST internally perform the following analytical steps for EACH
(source_id, query_id) pair before assigning a label:

1. Identify shared key words or phrases, focusing on distinctive or non-generic terms.
2. Examine whether the shared elements occur in a similar syntactic or structural order.
3. Check for slight morphological variation in shared words (e.g., different endings, cases, tenses).
4. Detect longer shared phrases or idiomatic / phraseological expressions that may indicate intentional reuse.

Use these definitions when making your decision:
- Text reuse: Direct reuse of wording from one text in another, typically with high lexical overlap.
- Intertextuality: A broader relationship where an author draws upon, references, or responds to another text,
  possibly through paraphrase, allusion, or stylistic imitation.

Given the following chunk from the query document:

{query_block}

And the following chunk from the source document:

{source_block}

Evaluate EACH possible (source_id, query_id) pair independently, even if some are not intertextual.
Also use the surrounding chunk context to inform your judgment.

IMPORTANT:
- Include  ONLY those (source_id, query_id) pairs
  for which an intertextual relationship is present.
- If an intertextual relationship is present, assign label 1.
- DO NOT include non-intertextual pairs.
- If no intertextual pairs exist, return an empty results list.
- Perform all reasoning internally.

Return ONLY a valid JSON object in the specified schema.
""".strip()

#Prompt register -> zero shot and few shot
PROMPT_BUILDERS: Dict[str,callable] = {
    "zero_shot_minimal": build_zero_shot_minimal_prompt,
    "zero_shot_medium": build_zero_shot_medium_prompt,
    "zero_shot_maximum": build_zero_shot_maximum_prompt,
}

# ================== BUILD FEW-SHOT PROMPT ==================
def format_few_shot_examples(
    examples: list[tuple[list[TextSegment], list[TextSegment], list[dict]]]
) -> str:
    blocks = []
    for q_window, s_window, results in examples:
        q_block = "\n".join(f"[{q.id}] {q.text}" for q in q_window)
        s_block = "\n".join(f"[{s.id}] {s.text}" for s in s_window)
        out = json.dumps({"results": results}, indent=2)

        blocks.append(
            f"""Query window:
{q_block}

Source window:
{s_block}

Output:
{out}"""
        )
    return "\n\n".join(blocks)

def build_few_shot_prompt(
    *,
    zero_shot_builder: callable,
    query_window: List[TextSegment],
    source_window: List[TextSegment],
    few_shot_samples: list[tuple[list[TextSegment], list[TextSegment], list[dict]]],
) -> str:
    zero_prompt = zero_shot_builder(query_window, source_window)

    marker = "Given the following chunk from the query document:"

    if marker not in zero_prompt:
        # Fallback: prepend examples at top
        return (
            "Here are some examples:\n\n"
            + format_few_shot_examples(few_shot_samples)
            + "\n\n"
            + zero_prompt
        )

    idx = zero_prompt.index(marker)

    few_shot_block = (
        "Here are some examples:\n\n"
        + format_few_shot_examples(few_shot_samples)
        + "\n\n"
    )

    return zero_prompt[:idx] + few_shot_block + zero_prompt[idx:]


# ================= PIPELINE =================

class LLMOnlySlidingWindowPipeline:
    
    """
    LLM-only intertextuality detection using
    fixed-size sliding windows over query and source.
    """

    def __init__(
        self,
        *,
        openai_model: str = "gpt-4o-mini",
        prompt_type: str = "zero_shot_minimal",
        chunk_size: int = 3,
        num_few_shot_examples: int=0,
        overlap: int=0, #how many rows overlap we want
    ):
        self.openai_model = openai_model
        self.prompt_builder = PROMPT_BUILDERS[prompt_type]
        self.chunk_size = chunk_size
        self.num_few_shot_examples = num_few_shot_examples
        self.overlap = overlap
        
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        
        self.stride = chunk_size - overlap
    
    # ---------- LLM WINDOW PREDICTION ----------
    def predict_window(
        self,
        source_window: List[TextSegment],
        query_window: List[TextSegment],
    ) -> List[dict]:
        
        #sample few-shot examples
        if self.num_few_shot_examples > 0:
            few_shot_samples = random.sample(
                FEW_SHOT_EXAMPLES,
                min(self.num_few_shot_examples, len(FEW_SHOT_EXAMPLES))
            )
            prompt = build_few_shot_prompt(zero_shot_builder=self.prompt_builder, query_window=query_window, source_window=source_window, few_shot_samples=few_shot_samples)
        else:
            prompt = self.prompt_builder(query_window, source_window)

        #prompt = self.prompt_builder(query_window, source_window)
        
        if self.openai_model.startswith("gpt-5"):
            response = client.responses.create(
                model=self.openai_model,
                reasoning={"effort": "medium"},
                input=prompt,
                text={
                    "format":{
                        "type": "json_schema",
                        "name": "intertextuality",
                        "schema": INTEXTUALITY_SCHEMA["schema"] 
                    }
                },
            )
            raw = response.output_text
            parsed = json.loads(raw)
        else:
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": INTEXTUALITY_SCHEMA
                },
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw)

        return parsed["results"]
    # ---------- FULL RUN ----------

    def run(
        self,
        *,
        query: Document,
        source: Document,
        checkpoint_path: str | None = None,
        resume: bool = False,
    ) -> FinalResults:
        """
        Run sliding-window intertextuality detection.

        Sliding strategy:
        - Fix source window
        - Slide query window completely
        - Then advance source window
        """
        chunk_size = self.chunk_size
        final_results: FinalResults = {}
        
        start_source_idx = 0
        if resume and checkpoint_path and os.path.exists(checkpoint_path):
            start_source_idx, final_results = load_partial_results(checkpoint_path)
            print(f"Resuming from source window {start_source_idx}")

        source_segments = pad_segments(
            list(source.segments.values()),
            self.chunk_size,
            prefix="S"
        )
        query_segments = pad_segments(
            list(query.segments.values()),
            self.chunk_size,
            prefix="Q"
        )

        source_windows = fixed_sliding_windows(source_segments, chunk_size, self.stride)
        query_windows = fixed_sliding_windows(query_segments, chunk_size, self.stride)

        if not source_windows or not query_windows:
            return final_results
        
        total_source_windows = len(source_windows)
        print(f"Total source windows: {total_source_windows}")

        for sw_idx, sw in enumerate(
            tqdm(source_windows[start_source_idx:], 
                desc="Processing source windows",
                initial=start_source_idx,
                total=total_source_windows), 
            start=start_source_idx
        ):
            for qw in query_windows:
                window_results = self.predict_window(sw, qw)
                
                # all expected pairs in current window (incl. padding)
                expected_pairs = [
                    (s.id, q.id)
                    for s in sw
                    for q in qw
                ]
                
                # positive pairs from LLM
                positive_pairs = {
                    (r["source_id"], r["query_id"])
                    for r in window_results
                    if "source_id" in r and "query_id" in r
                }
                """
                Padding segments are excluded from final_results,
                but still contribute to window context.
                """
                # aggregation across windows
                for source_id, query_id in expected_pairs:
                    # do not store padding pairs
                    if source_id.startswith("__PAD__"):
                        continue
                    if query_id.startswith("__PAD__"):
                        continue
                    label = 1 if (source_id, query_id) in positive_pairs else 0
                    key = (source_id, query_id)
                    final_results.setdefault(key, []).append(label)
                    
            if checkpoint_path:
                save_partial_results(
                    final_results,
                    checkpoint_path,
                    source_window_idx=sw_idx+1,
                )
        return final_results

# ================= PRETTY PRINT =================

def pretty_print(full_results: FullDict) -> None:
    """Print LLM Sliding-Window results in evaluator/pipeline format."""
    for query_id, lst in full_results.items():
        print(f"\n▶ Query segment {query_id!r}:")
        if not lst:
            print("  No intertextual links detected")
            continue
        for src_seg, sim, prob in lst:
            print(f"  {src_seg.id:<25} sim={sim:.3f} P(pos)={prob:.3f}")

# ================= MAIN =================

if __name__ == "__main__":

    query_doc = Document("../data/query.csv")
    source_doc = Document("../data/source.csv")

    pipeline = LLMOnlySlidingWindowPipeline(
        openai_model="gpt-5.4-nano",
        prompt_type="zero_shot_maximum",
        chunk_size=120,
        num_few_shot_examples=0,
        overlap=20
    )

    results = pipeline.run(
        query=query_doc,
        source=source_doc,
    )
    
    save_final_results(
        results,
        path="llm_results.json",
        meta={
            "model": pipeline.openai_model,
            "prompt_type": pipeline.prompt_builder.__name__,
            "chunk_size": pipeline.chunk_size,
            "num_few_shot_examples": pipeline.num_few_shot_examples,
            "overlap": pipeline.overlap,
        },
    )
    
    full_results = convert_llm_results(results, source_doc, threshold=0.5)
    #need method convert llm results coverage fair

    pretty_print(full_results)
