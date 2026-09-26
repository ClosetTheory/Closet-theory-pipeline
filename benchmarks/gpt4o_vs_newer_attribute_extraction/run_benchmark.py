"""GPT-4o vs newer vision models on Stage 3 attribute extraction. See RESULTS.md."""
import asyncio, base64, json, os, sys, time, re
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import httpx
from app.config import settings
from app.providers.vlm.openrouter import OpenRouterGPTProvider
from app.schemas.attributes import validate_extracted_attributes
SP = os.path.dirname(os.path.abspath(__file__))
import glob
images = sorted(glob.glob(os.path.join(REPO, "unknown_material_samples", "images", "*__raw.*")))[:24]
SKIP = {21}
GT = {
 1: ("dress", {"maxi_dress","dress","sundress"}, {"white","black","blue","multi","ivory","cream"}, {"print","graphic","abstract","patch","photo","novelty"}),
 2: ("top", {"tank_top","blouse","crop_top","tshirt"}, {"brown","rust","terracotta","orange","copper","cinnamon"}, {"solid","textur","plain"}),
 3: ("top|outerwear", {"blouse","kimono","button_down_shirt","blazer","dress_shirt","oxford_shirt","cardigan"}, {"white","multi","purple","cream","ivory"}, {"floral","print","abstract","watercol","botanic"}),
 4: ("dress", {"dress","maxi_dress","sundress"}, {"black","navy"}, {"floral","print","botanic"}),
 5: ("top", {"sweater","sweatshirt","tshirt"}, {"navy","black","blue"}, {"solid","strip","colour_block","color_block","colorblock","plain"}),
 6: ("top", {"blouse","tshirt","crop_top"}, {"white","cream","ivory","off-white","off_white"}, {"solid","plain","textur"}),
 7: ("top|dress", {"dress","blouse","tshirt","tank_top","maxi_dress","sundress","crop_top","skirt","midi_skirt"}, {"black","white","grey","gray"}, {"strip","geometric","abstract","print","graphic"}),
 8: ("top", {"vest","blouse","tank_top","crop_top"}, {"white","black","cream","ivory"}, {"solid","colour_block","color_block","colorblock","plain"}),
 9: ("top", {"tank_top","blouse","vest","crop_top","tshirt"}, {"black"}, {"solid","plain"}),
 10: ("bottom", {"trousers","chinos","dress_pants"}, {"black","charcoal"}, {"solid","plain"}),
 11: ("dress|top", {"dress","sundress","blouse","sweater","tshirt","maxi_dress"}, {"green","olive","forest","dark green","teal"}, {"solid","plain"}),
 12: ("bottom", {"leggings","sweatpants","trousers"}, {"red","coral","orange","scarlet","tomato"}, {"solid","plain"}),
 13: ("top", {"crop_top","blouse","tshirt","sweater"}, {"multi","black","red","green","white","blue","yellow"}, {"floral","print","graphic","patch","abstract","botanic"}),
 14: ("dress", {"dress","sundress","maxi_dress"}, {"black","beige","brown","gold","grey","gray","tan","taupe","cream"}, {"tweed","abstract","geometric","print","textur","houndstooth","check","plaid","boucl","graphic"}),
 15: ("top", {"button_down_shirt","oxford_shirt","dress_shirt","polo_shirt","blouse","tshirt","flannel_shirt"}, {"blue","light blue","white","sky","navy"}, {"strip","pinstrip"}),
 16: ("outerwear", {"blazer","suit_jacket","coat","leather_jacket","bomber_jacket","overcoat"}, {"black"}, {"solid","plain"}),
 17: ("top|outerwear", {"button_down_shirt","blouse","windbreaker","denim_jacket","oxford_shirt","dress_shirt","bomber_jacket","flannel_shirt"}, {"blue","light blue","white","sky","baby blue","powder"}, {"print","tie","abstract","cloud","graphic","ombre","marble"}),
 18: ("dress", {"dress","sundress"}, {"black","white"}, {"chevron","zigzag","zig","geometric","print","strip","herringbone"}),
 19: ("bottom", {"cargo_pants","trousers","sweatpants"}, {"black","charcoal","grey","gray","dark"}, {"solid","plain"}),
 20: ("bottom", {"palazzo_pants","trousers","dress_pants"}, {"blue","light blue","white","sky"}, {"strip","pinstrip"}),
 22: ("bottom", {"trousers","dress_pants","chinos"}, {"black","brown","dark","charcoal","espresso"}, {"solid","plain"}),
 23: ("bottom", {"shorts"}, {"red","orange","multi","scarlet"}, {"floral","print","botanic"}),
 24: ("bottom", {"leggings","sweatpants","trousers"}, {"coral","pink","salmon","orange","peach","red","watermelon"}, {"solid","plain"}),
}
MODELS = ["openai/gpt-4o","openai/gpt-6-luna","openai/gpt-6-sol","google/gemini-3.8-flash","google/gemini-3.5-flash-lite","anthropic/claude-sonnet-5","z-ai/glm-5.3-flash","qwen/qwen3.8-flash"]
BROAD = {"top":["top","shirt","blouse","sweater","knit","tee","tank","vest","cardigan","hoodie","polo"],
         "bottom":["bottom","pant","trouser","legging","short","skirt","jean","chino","cargo"],
         "dress":["dress","gown","jumpsuit","romper"],
         "outerwear":["outer","jacket","blazer","coat","parka"],
         "footwear":["shoe","foot","sneaker","boot","sandal","heel"],
         "accessory":["access","bag","belt","hat","scarf"]}
def broad(cat, sub):
    s = f"{cat} {sub}".lower()
    for b,kws in BROAD.items():
        if any(k in s for k in kws): return b
    return cat.lower()

async def capture_prompt():
    p = OpenRouterGPTProvider(api_key="x", base_url="http://127.0.0.1:1")
    try: await p.extract_attributes(b"\xff\xd8\xff", image_type=None, garment_label=None)
    except Exception: pass
    return p._last_prompt
def norm_sub(s): return re.sub(r"[\s\-]+","_",(s or "").strip().lower())

async def call(client, sem, model, idx, b64, prompt):
    payload = {"model": model, "messages":[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}],
               "max_tokens":1800,"temperature":0.1,"response_format":{"type":"json_object"},"usage":{"include":True}}
    headers={"Authorization":f"Bearer {settings.OPENROUTER_API_KEY}","Content-Type":"application/json","HTTP-Referer":"http://localhost:8000","X-Title":"Wardrobe Ingestion Pipeline"}
    err = "unknown"
    async with sem:
        for attempt in range(3):
            t=time.time()
            try:
                r = await client.post(f"{settings.OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload)
                dt=time.time()-t
                if r.status_code==429 or r.status_code>=500:
                    err=f"HTTP {r.status_code}: {r.text[:120]}"; await asyncio.sleep(4*(attempt+1)); continue
                if r.status_code!=200:
                    return {"model":model,"idx":idx,"ok":False,"err":f"HTTP {r.status_code}: {r.text[:150]}","latency":dt}
                j=r.json(); content=(j["choices"][0]["message"]["content"] or "").strip()
                m=re.search(r"\{.*\}",content,re.DOTALL); content=m.group(0) if m else content
                cost=(j.get("usage") or {}).get("cost")
                try:
                    attrs=validate_extracted_attributes(content); d=attrs.model_dump(mode="json"); valid=True; verr=None
                except Exception as e:
                    valid=False; verr=f"{type(e).__name__}: {str(e)[:160]}"
                    try: d=json.loads(content)
                    except Exception: d={}
                return {"model":model,"idx":idx,"ok":True,"valid":valid,"verr":verr,"latency":dt,"cost":cost,
                        "category":d.get("category"),"subcategory":d.get("subcategory"),"colour":d.get("colour"),"pattern":d.get("pattern"),"material":d.get("material"),"casual_name":d.get("casual_name")}
            except Exception as e:
                err=f"{type(e).__name__}: {str(e)[:120]}"
                await asyncio.sleep(2)
        return {"model":model,"idx":idx,"ok":False,"err":err,"latency":None}

async def main():
    prompt = await capture_prompt(); assert len(prompt)>500, "prompt capture failed"
    sem=asyncio.Semaphore(8); tasks=[]
    async with httpx.AsyncClient(timeout=120) as client:
        for i,path in enumerate(images, start=1):
            if i in SKIP: continue
            b64=base64.b64encode(open(path,"rb").read()).decode()
            for m in MODELS: tasks.append(call(client, sem, m, i, b64, prompt))
        results=await asyncio.gather(*tasks)
    json.dump(results, open(os.path.join(SP,"eval_results.json"),"w"), indent=1)
    n=len(GT)
    print(f"{'model':32s} {'valid':>6s} {'cat':>6s} {'subcat':>7s} {'colour':>7s} {'pattern':>8s} {'p50 s':>6s} {'$/img':>8s} errors")
    for m in MODELS:
        rs=[r for r in results if r["model"]==m]
        ok=[r for r in rs if r.get("ok")]; valid=sum(1 for r in ok if r["valid"])
        cat=sub=col=pat=0
        for r in ok:
            g=GT[r["idx"]]
            if broad(r.get("category") or "", r.get("subcategory") or "") in g[0].split("|"): cat+=1
            if norm_sub(r.get("subcategory")) in g[1]: sub+=1
            cols=r.get("colour") or []; cols=[cols] if isinstance(cols,str) else cols
            if any(any(a in (c or "").lower() for a in g[2]) for c in cols): col+=1
            if any(a in (r.get("pattern") or "").lower() for a in g[3]): pat+=1
        lat=sorted(r["latency"] for r in ok if r["latency"]); p50=lat[len(lat)//2] if lat else 0
        costs=[r["cost"] for r in ok if r.get("cost") is not None]; c=sum(costs)/len(costs) if costs else 0
        errs=len(rs)-len(ok)
        print(f"{m:32s} {valid:3d}/{n:<2d} {cat:3d}/{n:<2d} {sub:3d}/{n:<3d} {col:3d}/{n:<3d} {pat:3d}/{n:<4d} {p50:6.1f} {c:8.4f} {errs}")
    for r in results:
        if not r.get("ok"): print("ERR", r["model"], r["idx"], r["err"][:140])
asyncio.run(main())
