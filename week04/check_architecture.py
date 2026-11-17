"""
1주차: Encoder-only Transformer + Token Classification Head 구조 확인

사용법
  python model/check_architecture.py                          # klue/roberta-base
  python model/check_architecture.py --model xlm-roberta-base
  python model/check_architecture.py --compare                # 두 후보 비교표
  python model/check_architecture.py --scheme BIOES
  python model/check_architecture.py --random                 # 다운로드 없이 랜덤 가중치로 구조만 확인

확인하는 것
  1) Tokenizer가 Fast인지 (offset_mapping / word_ids 필요 → 문자 Span 복원용)
  2) 문장 → 토큰 → offset 정렬 모습
  3) 각 단계 텐서 shape: input_ids → hidden_states → attention → logits
  4) BIO/BIOES 라벨 수 = 엔티티 수에 따라 몇 개인지
"""
import argparse

import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
)

CANDIDATES = ["klue/roberta-base", "xlm-roberta-base"]
MVP_ENTITIES = ["PERSON", "PHONE", "EMAIL"]  # 29개 → 축약 매핑은 config/label_policy.yaml 합의 후 교체
SAMPLE = "홍길동 고객님 연락처는 010-1234-5678, 메일은 gildong@example.com 입니다."


def build_labels(entities, scheme="BIO"):
    prefixes = {"BIO": ["B", "I"], "BIOES": ["B", "I", "E", "S"]}[scheme]
    labels = ["O"] + [f"{p}-{e}" for e in entities for p in prefixes]
    return labels, {l: i for i, l in enumerate(labels)}, {i: l for i, l in enumerate(labels)}


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def load(model_name, num_labels, id2label, label2id, random_init):
    if random_init:
        # 다운로드 없이 roberta-base와 같은 크기의 구조만 생성 (가중치는 랜덤)
        cfg = RobertaConfig(vocab_size=32000, num_labels=num_labels,
                            id2label=id2label, label2id=label2id)
        cfg._attn_implementation = "eager"
        return None, RobertaForTokenClassification(cfg)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id,
        attn_implementation="eager")  # attention 가중치 꺼내보려면 eager 필요
    return tok, model


def inspect(model_name, scheme, random_init=False):
    labels, label2id, id2label = build_labels(MVP_ENTITIES, scheme)
    tok, model = load(model_name, len(labels), id2label, label2id, random_init)
    model.eval()
    cfg = model.config
    H, nH, L = cfg.hidden_size, cfg.num_attention_heads, cfg.num_hidden_layers

    print(f"\n{'=' * 70}\n모델: {'(random) ' if random_init else ''}{model_name}")
    print(f"layers={L}  hidden={H}  heads={nH}  head_dim={H // nH}  "
          f"ffn={cfg.intermediate_size}  vocab={cfg.vocab_size}  max_pos={cfg.max_position_embeddings}")
    enc_params = n_params(model.base_model)
    head_params = n_params(model.classifier)
    print(f"params: encoder={enc_params / 1e6:.1f}M  classifier={head_params:,} "
          f"(= {H}×{len(labels)} + {len(labels)})")
    print(f"embedding 비중: {n_params(model.base_model.embeddings) / enc_params:.0%}")

    # ---- 라벨 ----
    print(f"\n[{scheme}] 엔티티 {len(MVP_ENTITIES)}개 → 라벨 {len(labels)}개: {labels}")
    k = 2 if scheme == "BIO" else 4
    print(f"  공식: {k}×N + 1 (O)   | 29개 전부 쓰면 {k * 29 + 1}개")

    # ---- 입력 ----
    if tok is not None:
        print(f"\nFast tokenizer: {tok.is_fast}")
        enc = tok(SAMPLE, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
        word_ids = enc.word_ids()
        print(f"문장 길이 {len(SAMPLE)}자 → 토큰 {len(toks)}개")
        print(f"{'idx':>3} {'token':<14} {'offset':<10} {'word':>4}  text[s:e]")
        for i, (t, (s, e), w) in enumerate(zip(toks, offsets, word_ids)):
            print(f"{i:>3} {t:<14} {str((s, e)):<10} {str(w):>4}  {SAMPLE[s:e]!r}")
        print("※ special token은 offset (0,0) → 학습 시 라벨 -100")
    else:
        enc = {"input_ids": torch.randint(5, cfg.vocab_size, (2, 24)),
               "attention_mask": torch.ones(2, 24, dtype=torch.long)}

    # ---- forward ----
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, output_attentions=True)
    B, T = enc["input_ids"].shape
    print(f"\n[shape]  B=batch {B}, T=토큰 {T}, H={H}, heads={nH}, C=라벨 {len(labels)}")
    print(f"  input_ids         {tuple(enc['input_ids'].shape)}            (B, T)")
    print(f"  hidden_states     {len(out.hidden_states)}개 = embedding 1 + layer {L}")
    print(f"  last_hidden_state {tuple(out.hidden_states[-1].shape)}       (B, T, H)")
    print(f"  attention (1층)   {tuple(out.attentions[0].shape)}     (B, heads, T, T)")
    print(f"  logits            {tuple(out.logits.shape)}         (B, T, C)")
    s = out.attentions[0][0, 0].sum(-1)
    print(f"  attention 행 합 = {s.min():.3f}~{s.max():.3f} (softmax라 1)")
    pred = out.logits.argmax(-1)[0][:10].tolist()
    print(f"  argmax 예시(학습 전이라 무의미): {[id2label[i] for i in pred]}")
    return {"model": model_name, "layers": L, "hidden": H, "heads": nH, "vocab": cfg.vocab_size,
            "params(M)": round(n_params(model) / 1e6, 1), "tokens": T,
            "fast": getattr(tok, "is_fast", None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CANDIDATES[0])
    ap.add_argument("--scheme", default="BIO", choices=["BIO", "BIOES"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--random", action="store_true")
    a = ap.parse_args()

    rows = [inspect(m, a.scheme, a.random) for m in (CANDIDATES if a.compare else [a.model])]
    if len(rows) > 1:
        print(f"\n{'=' * 70}\n비교 (같은 문장)")
        keys = list(rows[0])
        print(" | ".join(f"{k:>18}" for k in keys))
        for r in rows:
            print(" | ".join(f"{str(r[k]):>18}" for k in keys))


if __name__ == "__main__":
    main()
