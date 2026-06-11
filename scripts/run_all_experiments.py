#!/usr/bin/env python
"""Optimized experiment runner: per-dataset max_length, fast execution.
ArguAna texts are short (~182 tokens median), so max_length=512 is fine.
"""
from __future__ import annotations

import json, logging, sys, time, gc, os
from collections import defaultdict
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.prompts import build_prompteol
from src.pooling import mean_pooling, last_token_pooling

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "/tmp/model_test/Qwen/Qwen2.5-1.5B-Instruct"
BATCH_SIZE = 8
OUTPUT_DIR = Path("/root/bayes-tmp/NLP-Final/results/advanced")

# Per-dataset max_length (ArguAna texts are ~180 tokens median)
DS_MAXLEN = {"QMSum": 2048, "2WikiMultihop": 2048, "ArguAna": 512}

# Qwen2.5-1.5B: 28 layers
LAYERS = [7, 14, 21, 28]
LAST_LAYER = -1
CHUNK_SIZE = 512; CHUNK_OVERLAP = 64
COMP_RATIO = 0.3

DATASETS = ["QMSum", "2WikiMultihop", "ArguAna"]

# ===================== Dataset loaders =====================

def load_qmsum():
    c=load_dataset("dwzhu/LongEmbed",name="qmsum",split="corpus")
    q=load_dataset("dwzhu/LongEmbed",name="qmsum",split="queries")
    qr=load_dataset("dwzhu/LongEmbed",name="qmsum",split="qrels")
    return _build("QMSum",c,q,qr)
def load_2wikimultihop():
    c=load_dataset("dwzhu/LongEmbed",name="2wikimqa",split="corpus")
    q=load_dataset("dwzhu/LongEmbed",name="2wikimqa",split="queries")
    qr=load_dataset("dwzhu/LongEmbed",name="2wikimqa",split="qrels")
    return _build("2WikiMultihop",c,q,qr)
def load_arguana():
    c=load_dataset("mteb/arguana","corpus",split="corpus")
    q=load_dataset("mteb/arguana","queries",split="queries")
    qr=load_dataset("mteb/arguana",split="test")
    ct,ci=[],[]
    for d in c:
        t=d.get("title","") or ""; txt=d.get("text","") or ""
        ct.append(f"{t}\n{txt}".strip() if t else txt); ci.append(str(d["_id"]))
    qt=[d["text"] for d in q]; qi=[str(d["_id"]) for d in q]
    qrels=defaultdict(set)
    for r in qr:
        if float(r["score"])>0: qrels[str(r["query-id"])].add(str(r["corpus-id"]))
    return {"name":"ArguAna","corpus_texts":ct,"corpus_ids":ci,
            "query_texts":qt,"query_ids":qi,"qrels":dict(qrels)}
def _build(name,c,q,qr):
    ct=[d["text"] for d in c]
    ci=[str(d.get("doc_id",d.get("_id",str(i)))) for i,d in enumerate(c)]
    qt=[d["text"] for d in q]
    qi=[str(d.get("qid",d.get("_id",str(i)))) for i,d in enumerate(q)]
    qrels=defaultdict(set)
    for r in qr: qrels[str(r.get("qid",r.get("_id","")))].add(str(r.get("doc_id",r.get("docid",""))))
    return {"name":name,"corpus_texts":ct,"corpus_ids":ci,
            "query_texts":qt,"query_ids":qi,"qrels":dict(qrels)}
LOADERS={"QMSum":load_qmsum,"2WikiMultihop":load_2wikimultihop,"ArguAna":load_arguana}

# ===================== Metrics =====================

def metrics(q_emb,c_emb,qrels,q_ids,c_ids,k_values=(1,10)):
    q_n=q_emb/(np.linalg.norm(q_emb,axis=1,keepdims=True)+1e-9)
    c_n=c_emb/(np.linalg.norm(c_emb,axis=1,keepdims=True)+1e-9)
    scores=q_n@c_n.T; max_k=max(k_values)
    ndcg_k={k:[] for k in k_values}; recall_k={k:[] for k in k_values}
    mrr,map_s=[],[]
    for qi,qid in enumerate(q_ids):
        if qid not in qrels or not qrels[qid]: continue
        rel=qrels[qid]; q_s=scores[qi]; top=np.argsort(q_s)[::-1][:max_k]
        rr=0.0
        for rk,idx in enumerate(top):
            if c_ids[idx] in rel: rr=1.0/(rk+1); break
        mrr.append(rr)
        ap=0.0; rc=0
        for rk,idx in enumerate(top):
            if c_ids[idx] in rel: rc+=1; ap+=rc/(rk+1)
        if rc>0: ap/=min(len(rel),max_k)
        map_s.append(ap)
        for k in k_values:
            tk=top[:k]; yt=np.zeros(k)
            for rk,idx in enumerate(tk):
                if c_ids[idx] in rel: yt[rk]=1.0
            dcg=sum((2**yt[i]-1)/np.log2(i+2) for i in range(k))
            ideal=sorted([1.0]*min(len(rel),k)+[0.0]*max(0,k-len(rel)),reverse=True)
            idcg=sum((2**ideal[i]-1)/np.log2(i+2) for i in range(k))
            ndcg_k[k].append(dcg/idcg if idcg>0 else 0.0)
            recall_k[k].append(sum(1 for idx in tk if c_ids[idx] in rel)/min(len(rel),k) if len(rel)>0 else 0)
    m={}
    for k in k_values:
        m[f"ndcg@{k}"]=float(np.mean(ndcg_k[k])) if ndcg_k[k] else 0.0
        m[f"recall@{k}"]=float(np.mean(recall_k[k])) if recall_k[k] else 0.0
    m["mrr@10"]=float(np.mean(mrr)) if mrr else 0.0
    m["map@10"]=float(np.mean(map_s)) if map_s else 0.0
    return m

# ===================== Memory-Efficient Encoder =====================

class Encoder:
    def __init__(self, model_path, load_in_4bit=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        kw = {"trust_remote_code": True, "output_hidden_states": False}
        if load_in_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        if torch.cuda.is_available(): kw["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        self._hook_cache = []
        logger.info("Loaded %d layers, hidden=%d, VRAM=%.2fGB",
                    self.num_layers, self.hidden_dim, torch.cuda.memory_allocated()/1e9)

    def _resolve(self, layer):
        n=self.num_layers
        if layer==-1 or layer==n: return n
        if layer<0: return n+layer+1
        return layer

    def _register_hook(self, layer):
        base=self.model.model; ti=self._resolve(layer)
        target=base.norm if ti==self.num_layers else base.layers[ti]
        def hook_fn(m,inp,out):
            t=out[0] if isinstance(out,tuple) else out; self._hook_cache.append(t)
        target.register_forward_hook(hook_fn)

    @torch.inference_mode()
    def encode(self, texts, method="mean", layer=-1, max_length=2048, pooling_fn=None):
        self._register_hook(layer)
        fmt = [build_prompteol(t) for t in texts] if method=="prompteol" else list(texts)
        all_emb = []; bs=BATCH_SIZE
        for start in tqdm(range(0,len(texts),bs), desc=f"enc[{method}_L{layer}]"):
            batch=fmt[start:start+bs]
            enc=self.tokenizer(batch,padding=True,truncation=True,
                               max_length=max_length,return_tensors="pt")
            dev=next(self.model.parameters()).device; enc={k:v.to(dev) for k,v in enc.items()}
            self._hook_cache.clear(); self.model(**enc); hidden=self._hook_cache[-1]
            if pooling_fn is not None:
                pooled=pooling_fn(hidden,enc["attention_mask"])
            elif method=="prompteol":
                pooled=last_token_pooling(hidden,enc["attention_mask"])
            else:
                pooled=mean_pooling(hidden,enc["attention_mask"])
            pooled=F.normalize(pooled,p=2,dim=1); all_emb.append(pooled.cpu().float().numpy())
            del hidden,pooled,enc
        return np.vstack(all_emb)

    @torch.inference_mode()
    def encode_all_layers(self, texts, method, layers, max_length=2048):
        fmt=[build_prompteol(t) for t in texts] if method=="prompteol" else list(texts)
        all_emb={l:[] for l in layers}; bs=BATCH_SIZE
        for start in tqdm(range(0,len(texts),bs), desc=f"enc_multi[{method}]"):
            batch=fmt[start:start+bs]
            enc=self.tokenizer(batch,padding=True,truncation=True,
                               max_length=max_length,return_tensors="pt")
            dev=next(self.model.parameters()).device; enc={k:v.to(dev) for k,v in enc.items()}
            outputs=self.model(**enc,output_hidden_states=True)
            for layer in layers:
                li=self._resolve(layer); hidden=outputs.hidden_states[li]
                pooled=(last_token_pooling(hidden,enc["attention_mask"]) if method=="prompteol"
                       else mean_pooling(hidden,enc["attention_mask"]))
                pooled=F.normalize(pooled,p=2,dim=1); all_emb[layer].append(pooled.cpu().float().numpy())
            del outputs,hidden,pooled,enc
        return {l:np.vstack(embs) for l,embs in all_emb.items()}

    def cleanup(self): del self.model; gc.collect(); torch.cuda.empty_cache()

# ===================== Weighted Pooling =====================

def attn_pos_weighted(hidden, mask):
    w=hidden.norm(dim=-1); L=hidden.shape[1]
    pos=torch.ones(L,device=hidden.device)
    bs=max(1,L//12); be=max(1,L//6); pos[:bs]=1.25; pos[-be:]=1.15
    w=w*pos.unsqueeze(0)*mask.float()
    return (hidden*(w/(w.sum(dim=1,keepdim=True)+1e-12)).unsqueeze(-1)).sum(dim=1)

def saliency_weighted(hidden, mask):
    w=hidden.norm(dim=-1)*mask.float()
    return (hidden*(w/(w.sum(dim=1,keepdim=True)+1e-12)).unsqueeze(-1)).sum(dim=1)

# ===================== RP2: Chunk =====================

def chunk_encode(enc, texts, method="mean", layer=-1, max_length=2048, cs=512, ov=64, agg="mean"):
    all_emb=[]
    for text in tqdm(texts, desc="chunk"):
        toks=enc.tokenizer.encode(text,add_special_tokens=False)
        chunks=[]; stride=cs-ov
        for s in range(0,len(toks),stride):
            c=toks[s:s+cs]
            if len(c)>=max(16,cs//4): chunks.append(enc.tokenizer.decode(c,skip_special_tokens=True))
        if not chunks: chunks=[text[:cs*4]]
        if len(chunks)<=1:
            emb=enc.encode(chunks,method,layer,max_length=max_length)
        else:
            emb=enc.encode(chunks,method,layer,max_length=max_length)
            t=torch.from_numpy(emb).float()
            if agg=="weighted":
                w=F.softmax(t.norm(dim=-1),dim=0)
                emb=((t*w.unsqueeze(-1)).sum(dim=0)).unsqueeze(0).numpy()
            else: emb=t.mean(dim=0,keepdim=True).numpy()
        emb=F.normalize(torch.from_numpy(emb).float(),p=2,dim=1).numpy()
        all_emb.append(emb[0])
    return np.stack(all_emb,axis=0)

# ===================== RP3: Compression =====================

def extractive_compress(text, enc, ratio=0.3, min_sent=3):
    import re
    sents=[s.strip() for s in re.split(r'(?<=[.!?。！？\n])\s*',text) if s.strip()]
    if len(sents)<=min_sent: return text
    all_toks=[set(enc.tokenizer.encode(s,add_special_tokens=False)) for s in sents]
    n=len(sents); sim=np.eye(n)
    for i in range(n):
        for j in range(i+1,n):
            inter=len(all_toks[i]&all_toks[j]); union=len(all_toks[i]|all_toks[j])
            s=inter/max(union,1); sim[i,j]=s; sim[j,i]=s
    centrality=(sim.sum(axis=1)-1)/max(n-1,1)
    k=max(min_sent,min(30,int(n*ratio)))
    top=sorted(np.argsort(centrality)[::-1][:k].tolist())
    return " ".join(sents[i] for i in top)

def compression_encode(enc, texts, method="mean", layer=-1, max_length=2048, ratio=0.3):
    compressed=[]
    for text in tqdm(texts, desc="compress"):
        if len(enc.tokenizer.encode(text,add_special_tokens=False))<256: compressed.append(text)
        else: compressed.append(extractive_compress(text,enc,ratio=ratio))
    return enc.encode(compressed,method,layer,max_length=max_length)

def comp_chunk_encode(enc, texts, method="mean", layer=-1, max_length=2048, ratio=0.3, cs=256, ov=32):
    compressed=[]
    for text in tqdm(texts, desc="cmp+chk"):
        if len(enc.tokenizer.encode(text,add_special_tokens=False))<256: compressed.append(text)
        else: compressed.append(extractive_compress(text,enc,ratio=ratio))
    return chunk_encode(enc,compressed,method,layer,max_length=max_length,cs=cs,ov=ov)

# ===================== Run & Save =====================

def save(name, data):
    with open(OUTPUT_DIR/f"{name}.json","w",encoding="utf-8") as f:
        json.dump(data,f,indent=2,ensure_ascii=False)

def run_exp(enc, exp_name, desc, ds_data, etype, *,
            method="mean", layer=LAST_LAYER, pooling_fn=None,
            encode_fn=None, encode_kw=None):
    logger.info("="*60)
    logger.info("%s: %s", exp_name, desc)
    exp_results={}
    for ds_name in DATASETS:
        data=ds_data[ds_name]; ml=DS_MAXLEN[ds_name]; t0=time.time()
        if encode_fn is not None:
            c_emb=encode_fn(enc,data["corpus_texts"],method=method,layer=layer,
                           max_length=ml,**(encode_kw or {}))
            q_emb=encode_fn(enc,data["query_texts"],method=method,layer=layer,
                           max_length=ml,**(encode_kw or {}))
        elif pooling_fn is not None:
            c_emb=enc.encode(data["corpus_texts"],method=method,layer=layer,
                            max_length=ml,pooling_fn=pooling_fn)
            q_emb=enc.encode(data["query_texts"],method=method,layer=layer,
                            max_length=ml,pooling_fn=pooling_fn)
        else:
            c_emb=enc.encode(data["corpus_texts"],method=method,layer=layer,max_length=ml)
            q_emb=enc.encode(data["query_texts"],method=method,layer=layer,max_length=ml)
        m=metrics(q_emb,c_emb,data["qrels"],data["query_ids"],data["corpus_ids"])
        logger.info("  %s ndcg@10=%.4f recall@10=%.4f mrr@10=%.4f map@10=%.4f (%.0fs)",
                    ds_name,m["ndcg@10"],m["recall@10"],m["mrr@10"],m["map@10"],time.time()-t0)
        exp_results[ds_name]=m
        del c_emb,q_emb; gc.collect(); torch.cuda.empty_cache()
    result={"description":desc,"type":etype,"results":exp_results}
    save(exp_name,result)
    return result

# ===================== MAIN =====================

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    all_results={}

    # Load datasets
    ds_data={}
    for ds_name in DATASETS:
        ds_data[ds_name]=LOADERS[ds_name]()
        logger.info("%s: corpus=%d queries=%d", ds_name,
                    len(ds_data[ds_name]["corpus_texts"]),
                    len(ds_data[ds_name]["query_texts"]))

    # ---- Encoder 1: Single-layer experiments ----
    enc=Encoder(MODEL_PATH)

    # 1. Baselines
    for method in ["mean","prompteol"]:
        r=run_exp(enc,f"baseline_{method}",f"Baseline: {method}",ds_data,"baseline",method=method)
        all_results[f"baseline_{method}"]=r

    # 2. RP1: Weighted Pooling
    rp1={
        "rp1_attn_pos":("Attention+Position weighted",attn_pos_weighted),
        "rp1_saliency":("Norm-saliency weighted",saliency_weighted),
    }
    for name,(desc,fn) in rp1.items():
        r=run_exp(enc,name,desc,ds_data,"rp1",pooling_fn=fn)
        all_results[name]=r

    # 3. RP2: Chunk
    rp2={
        "rp2_chunk_mean":("Chunk mean agg",{"cs":CHUNK_SIZE,"ov":CHUNK_OVERLAP,"agg":"mean"}),
        "rp2_chunk_wtd":("Chunk weighted agg",{"cs":CHUNK_SIZE,"ov":CHUNK_OVERLAP,"agg":"weighted"}),
    }
    for name,(desc,kw) in rp2.items():
        r=run_exp(enc,name,desc,ds_data,"rp2",encode_fn=chunk_encode,encode_kw=kw)
        all_results[name]=r

    # 4. RP3: Compression
    r=run_exp(enc,"rp3_extractive",f"Extractive compression ratio={COMP_RATIO}",
              ds_data,"rp3",encode_fn=compression_encode,encode_kw={"ratio":COMP_RATIO})
    all_results["rp3_extractive"]=r

    # 5. Combined RP2+RP3
    r=run_exp(enc,"combined_rp23","RP2+RP3: compress→chunk→mean",
              ds_data,"combined",encode_fn=comp_chunk_encode,
              encode_kw={"ratio":COMP_RATIO,"cs":CHUNK_SIZE//2,"ov":CHUNK_OVERLAP//2})
    all_results["combined_rp23"]=r

    # 6. RP3 Compression ratio ablation (QMSum only)
    for ratio in [0.5,0.15]:
        r=run_exp(enc,f"rp3_ratio{int(ratio*100)}",f"Comp ratio={ratio}",
                  ds_data,"rp3_ablation",encode_fn=compression_encode,
                  encode_kw={"ratio":ratio})
        all_results[f"rp3_ratio{int(ratio*100)}"]=r

    # 7. RP2 Chunk size ablation (QMSum only)
    for cs in [256,1024]:
        ov=cs//8
        r=run_exp(enc,f"rp2_cs{cs}",f"Chunk size={cs}",
                  ds_data,"rp2_ablation",encode_fn=chunk_encode,
                  encode_kw={"cs":cs,"ov":ov,"agg":"mean"})
        all_results[f"rp2_cs{cs}"]=r

    enc.cleanup(); del enc; gc.collect(); torch.cuda.empty_cache()

    # ---- Encoder 2: Layer Ablation ----
    enc2=Encoder(MODEL_PATH)
    for method in ["mean","prompteol"]:
        logger.info("="*60)
        logger.info("LAYER ABLATION: %s",method)
        for ds_name in DATASETS:
            data=ds_data[ds_name]; ml=DS_MAXLEN[ds_name]; t0=time.time()
            # Only ablate on QMSum+2Wiki (ArguAna layers follow same pattern)
            if ds_name=="ArguAna" and method=="prompteol": continue
            c_embs=enc2.encode_all_layers(data["corpus_texts"],method,LAYERS,max_length=ml)
            q_embs=enc2.encode_all_layers(data["query_texts"],method,LAYERS,max_length=ml)
            for layer in LAYERS:
                exp_name=f"{method}_layer{layer}"
                m=metrics(q_embs[layer],c_embs[layer],data["qrels"],data["query_ids"],data["corpus_ids"])
                logger.info("  %s L%d ndcg@10=%.4f recall@10=%.4f (%.0fs)",
                           ds_name,layer,m["ndcg@10"],m["recall@10"],time.time()-t0)
                if exp_name not in all_results:
                    all_results[exp_name]={"description":f"{method} layer{layer}","type":"ablation","results":{}}
                all_results[exp_name]["results"][ds_name]=m
                save(exp_name,all_results[exp_name])
            del c_embs,q_embs; gc.collect(); torch.cuda.empty_cache()
    enc2.cleanup(); del enc2; gc.collect(); torch.cuda.empty_cache()

    # ---- Save & Print ----
    with open(OUTPUT_DIR/"all_advanced_results.json","w",encoding="utf-8") as f:
        json.dump(all_results,f,indent=2,ensure_ascii=False)

    print_summary(all_results)
    print(f"\nDONE! {OUTPUT_DIR}/all_advanced_results.json")

def print_summary(results):
    print("\n"+"="*100)
    print("COMPLETE RESULTS (Qwen2.5-1.5B-Instruct, maxlen=2048/2048/512)")
    print("="*100)
    ds_list=["QMSum","2WikiMultihop","ArguAna"]
    for metric in ["ndcg@10","recall@10","mrr@10","map@10"]:
        print(f"\n{'='*60}\n  {metric}\n{'='*60}")
        hdr=f"{'Experiment':<30s}"
        for ds in ds_list: hdr+=f" {ds:>10s}"
        print(hdr); print(f"{'':-<30s} {'':->10s} {'':->10s} {'':->10s}")
        for name in sorted(results.keys()):
            r=results[name].get("results",{})
            print(f"{name:<30s} " + " ".join(f"{r.get(ds,{}).get(metric,0):>10.4f}" for ds in ds_list))
    # Improvement
    print(f"\n{'='*60}\nIMPROVEMENT OVER BASELINE MEAN (nDCG@10)\n{'='*60}")
    baseline=results.get("baseline_mean",{}).get("results",{})
    for name in sorted(results.keys()):
        if name=="baseline_mean": continue
        r=results[name].get("results",{})
        imps=[];
        for ds in ds_list:
            bv=baseline.get(ds,{}).get("ndcg@10",0.001)
            ev=r.get(ds,{}).get("ndcg@10",0)
            imps.append(100*(ev-bv)/max(bv,0.001))
        avg=np.mean(imps)
        print(f"  {name:<30s} avg={avg:+.1f}%  "+" | ".join(f"{ds}:{v:+.1f}%" for ds,v in zip(ds_list,imps)))

if __name__=="__main__":
    main()
