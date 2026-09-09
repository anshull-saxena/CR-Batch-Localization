import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from cr_batcher import CRBatcher

np.random.seed(42)

def run_stress_benchmark():
    templates = [
        "Schedule the best deals on {target}",
        "Manage customer reviews on {target}",
        "Failed to initialize connection to database cluster {target} with error code {code}",
        "Are you sure you want to permanently delete the selected item from {target}?",
        "An unhandled exception occurred while processing the telemetry event: {target} at stack frame {code}.",
        "Save", "Cancel", "Next", "Back", "Options", "Settings", "Help"
    ]
    
    segments = []
    for i in range(500):
        t = np.random.choice(templates)
        text = t.format(target=f"Service_{i}", code=f"0x{i:04X}")
        if np.random.rand() < 0.1:
            text += " " + ("This software and documentation are confidential and proprietary. " * 4)
        segments.append((i, text))

    languages = ["de-DE", "ru-RU", "es-ES", "zh-CN", "ja-JP", "hi-IN"]
    print(f"=== Multilingual Enterprise Stress Benchmark (500 Segments) ===\n")
    header = f"{'Language':8} | {'Metric':24} | {'Fixed 52':12} | {'Static 512':12} | {'CR-Batch':12} | {'Reduction':10}"
    print(header)
    print("-" * len(header))

    for lang in languages:
        b1 = [segments[i:i+52] for i in range(0, len(segments), 52)]
        b2 = []
        curr = []
        c_tok = 0
        for idx, txt in segments:
            tok = max(1, int(len(txt) / 4.0))
            if c_tok + tok > 512 and curr:
                b2.append(curr)
                curr = [(idx, txt)]
                c_tok = tok
            else:
                curr.append((idx, txt))
                c_tok += tok
        if curr:
            b2.append(curr)
            
        batcher = CRBatcher(token_budget=512, max_batch_items=52)
        b3 = batcher.schedule(segments, target_lang=lang)

        def get_waste(batches, target_lang):
            enc_pad, enc_real = 0, 0
            q90 = batcher.EXPANSION_PROFILES.get(target_lang, batcher.DEFAULT_PROFILE)["q90"]
            for b in batches:
                s_lens = [max(1, int(len(t) / 4.0)) for _, t in b]
                max_s = max(s_lens)
                enc_pad += sum(max_s - s for s in s_lens)
                enc_real += sum(s_lens)
            return 100.0 * enc_pad / (enc_pad + enc_real)

        w1 = get_waste(b1, lang)
        w2 = get_waste(b2, lang)
        w3 = get_waste(b3, lang)
        red = (w2 - w3) / w2 * 100
        print(f"{lang:8} | {'Encoder Padding Waste':24} | {w1:9.2f}% | {w2:9.2f}% | {w3:9.2f}% | {red:8.1f}%")

if __name__ == "__main__":
    run_stress_benchmark()
