"""
make_report.py — 整合 out/<dataset>/ 下的結果與圖, 產生單一深色 HTML 報告 (圖片內嵌, 可直接分享)

    python scripts/make_report.py                     # LBNL
    python scripts/make_report.py --dataset limassol
"""
import argparse
import base64
import html
import json
from datetime import datetime
from pathlib import Path

from hvac.datasets import get_spec


def img(fig_dir, name, caption):
    p = fig_dir / name
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}">'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')


def f3(x): return f"{x:.3f}"
def f4(x): return f"{x:.4f}"
def f2(x): return f"{x:.2f}"
def pm(d, fmt=f3): return f"{fmt(d['mean'])} ± {fmt(d['std'])}"
def pct(new, base): return 100 * (base - new) / base
def cls(v): return "good" if v > 0 else "bad"


def arch_svg(spec, L):
    hist = "、".join(c.replace("_c", "").replace("_", " ") for c in spec.seq_cols if c not in ("h_sin", "h_cos"))
    now = "、".join(c.replace("_c", "").replace("_", " ") for c in spec.now_cols if c not in ("h_sin", "h_cos"))
    rf = 1 + 2 * (1 + 2 + 4)
    return f"""
<svg viewBox="0 0 760 250" class="diagram" role="img" aria-label="CNN_MLP 架構圖">
 <defs><marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
  <path d="M0,0 L8,4 L0,8 z" fill="#8b949e"/></marker></defs>
 <g font-size="12" fill="#e6edf3" text-anchor="middle">
  <rect x="10" y="20" width="160" height="70" rx="8" class="box in"/>
  <text x="90" y="42">過去 {L} 步（{spec.steps_to_h(L):g} h）</text>
  <text x="90" y="62" class="sm">{html.escape(hist)}</text><text x="90" y="78" class="sm">+ 時間 sin/cos</text>
  <rect x="205" y="20" width="150" height="70" rx="8" class="box cnn"/>
  <text x="280" y="45">Conv1d × 3</text><text x="280" y="63" class="sm">dilation 1 / 2 / 4</text>
  <text x="280" y="78" class="sm">感受野 {rf} 步</text>
  <rect x="390" y="20" width="110" height="70" rx="8" class="box cnn"/>
  <text x="445" y="50">Flatten</text><text x="445" y="68">FC 64</text>
  <rect x="10" y="150" width="160" height="70" rx="8" class="box in"/>
  <text x="90" y="172">下一步輸入</text><text x="90" y="192" class="sm">{html.escape(now)}</text>
  <text x="90" y="208" class="sm">+ 時間 sin/cos</text>
  <rect x="205" y="150" width="150" height="70" rx="8" class="box mlp"/><text x="280" y="190">MLP 32</text>
  <rect x="540" y="85" width="90" height="70" rx="8" class="box head"/>
  <text x="585" y="115">Concat</text><text x="585" y="133">FC 64 → 2</text>
  <rect x="660" y="60" width="90" height="45" rx="8" class="box out"/><text x="705" y="87">ΔT 室溫</text>
  <rect x="660" y="135" width="90" height="45" rx="8" class="box out"/><text x="705" y="162">P 功率</text>
 </g>
 <g stroke="#8b949e" stroke-width="1.5" fill="none" marker-end="url(#ar)">
  <path d="M170,55 H203"/><path d="M355,55 H388"/><path d="M500,55 H520 V110 H538"/>
  <path d="M170,185 H203"/><path d="M355,185 H520 V130 H538"/>
  <path d="M630,112 H645 V82 H658"/><path d="M630,128 H645 V157 H658"/>
 </g>
</svg>"""



CSS = """
:root{--bg:#0d1117;--card:#161b22;--line:#30363d;--fg:#e6edf3;--mut:#8b949e;--acc:#4fc3f7;--ok:#81c784;--bad:#e57373;--warn:#ffb74d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font-family:"Microsoft JhengHei","PingFang TC","Noto Sans TC",sans-serif;line-height:1.65}
header{padding:28px 32px 12px;border-bottom:1px solid var(--line)}h1{margin:0;font-size:24px}header p{margin:4px 0 0;color:var(--mut)}
nav{display:flex;flex-wrap:wrap;gap:6px;padding:12px 32px;position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);z-index:5}
nav button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 14px;cursor:pointer;font:inherit;font-size:14px}
nav button.on{border-color:var(--acc);color:var(--acc)}main{padding:20px 32px 60px;max-width:1200px}
section{display:none}section.on{display:block}h2{font-size:20px;border-left:4px solid var(--acc);padding-left:10px}h3{font-size:16px;color:var(--acc)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:12px 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}.card .k{color:var(--mut);font-size:13px}
.card .v{font-size:26px;font-weight:700;margin:2px 0}.card .s{font-size:13px}.good{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
table{border-collapse:collapse;width:100%;margin:10px 0 18px;font-size:14px}th,td{border:1px solid var(--line);padding:6px 10px;text-align:center}
th{background:var(--card)}td:first-child,th:first-child{text-align:left}tr.hl td{background:#1c2a36}
figure{margin:16px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}figure img{width:100%;display:block;border-radius:6px}
figcaption{color:var(--mut);font-size:13px;margin-top:6px}.muted{color:var(--mut)}
.note{background:var(--card);border-left:4px solid var(--warn);padding:10px 14px;border-radius:6px;margin:12px 0}
.diagram{width:100%;max-width:820px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px}
.box{stroke-width:1.5}.in{fill:#1f2937;stroke:#8b949e}.cnn{fill:#0f2a3a;stroke:#4fc3f7}.mlp{fill:#33270f;stroke:#ffb74d}.head{fill:#1d3020;stroke:#81c784}.out{fill:#2a1a1a;stroke:#e57373}
.sm{font-size:10.5px;fill:#8b949e}ul{padding-left:22px}code{background:#1f2937;padding:1px 6px;border-radius:4px}
"""

JS = """
const bs=document.querySelectorAll('nav button'),ss=document.querySelectorAll('section');
bs.forEach((b,i)=>b.onclick=()=>{bs.forEach(x=>x.classList.remove('on'));ss.forEach(x=>x.classList.remove('on'));b.classList.add('on');ss[i].classList.add('on');window.scrollTo(0,0)});
"""



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lbnl", choices=["lbnl", "limassol"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    spec = get_spec(a.dataset)
    out = Path(a.out or f"out/{a.dataset}"); fig = out / "figures"
    M = json.loads((out / "results" / "main.json").read_text(encoding="utf-8"))
    load = lambda n: json.loads((out / "results" / n).read_text(encoding="utf-8")) if (out / "results" / n).exists() else None
    E, D = load("eval.json"), load("data_check.json")
    cfg = M["config"]; L, H = cfg["L"], cfg["H"]
    hz = lambda k: f"{k * spec.dt_min} 分鐘" if spec.dt_min < 60 else f"{k * spec.dt_min // 60} 小時"
    k_mid = max(1, H // 4)
    cnn, mlp, per = M["summary"]["CNN_MLP"], M["summary"]["MLPOnly"], M["persistence"]
    ce, me = cnn["ensemble"], mlp["ensemble"]
    rc, rm, rp = ce["rollout"][-1], me["rollout"][-1], per["rollout"][-1]

    # ─── 總覽卡片 ───
    cards = f"""<div class="cards">
 <div class="card"><div class="k">單步室溫 RMSE（CNN 集成）</div><div class="v">{f4(ce['T_rmse'])} °C</div>
  <div class="s {cls(pct(ce['T_rmse'], per['T_rmse']))}">vs persistence {f4(per['T_rmse'])}（{pct(ce['T_rmse'], per['T_rmse']):+.1f}%）</div></div>
 <div class="card"><div class="k">{hz(H)} rollout 室溫 RMSE</div><div class="v">{f3(rc)} °C</div>
  <div class="s {cls(pct(rc, rp))}">vs persistence {f3(rp)}（{pct(rc, rp):+.1f}%）</div></div>
 <div class="card"><div class="k">單步功率 RMSE（CNN 集成）</div><div class="v">{f2(ce['P_rmse'])} kW</div>
  <div class="s {cls(pct(ce['P_rmse'], me['P_rmse']))}">vs MLP-only {f2(me['P_rmse'])}（{pct(ce['P_rmse'], me['P_rmse']):+.1f}%）</div></div>"""
    if "T_violation" in ce:
        cards += f"""
 <div class="card"><div class="k">物理方向違反率（控制量 → 室溫）</div><div class="v">{ce['T_violation']:.1%}</div>
  <div class="s {'good' if ce['T_violation'] < 0.05 else 'warn'}">單調約束 λ = {cfg['mono']:g}</div></div>"""
    cards += "</div>"

    # ─── 結論 ───
    concl = [
        f"Day 2 完成標準：1D-CNN+MLP 與 MLP-only 皆已訓練（各 {cfg['seeds']} 個 seed），"
        f"預測 vs 實際見圖 2，多步 rollout 誤差見圖 4。",
        f"單步室溫：CNN 集成 {f4(ce['T_rmse'])} °C、MLP 集成 {f4(me['T_rmse'])} °C、persistence {f4(per['T_rmse'])} °C。",
        f"多步室溫（控制量用實際紀錄）：{hz(k_mid)} 時 CNN {f3(ce['rollout'][k_mid - 1])}、MLP {f3(me['rollout'][k_mid - 1])}、"
        f"persistence {f3(per['rollout'][k_mid - 1])} °C；{hz(H)} 時 CNN {f3(rc)}、MLP {f3(rm)}、persistence {f3(rp)} °C。",
        f"功率：CNN 集成 {f2(ce['P_rmse'])} kW，" + ("優於" if ce['P_rmse'] < me['P_rmse'] else "未優於")
        + f" MLP-only 的 {f2(me['P_rmse'])} kW（測試集功率標準差 {f2(ce['P_std'])} kW）。",
    ]
    if E and "constraint" in E:
        c = E["constraint"]
        sf, sa = E["constraint"]["sweep_free"], E["sweep"]["all"]
        concl.append(
            f"物理約束：未加約束的模型有 {c['free'].get('T_violation', 0):.0%} 的測試樣本呈現「閥門開大 → 室溫上升」，"
            f"閥門 0→1 時室溫 {sf['T'][0]:.1f}→{sf['T'][-1]:.1f} °C（違反物理）；加約束後違反率 "
            f"{c['constrained'].get('T_violation', 0):.0%}，室溫 {sa['T'][0]:.1f}→{sa['T'][-1]:.1f} °C。"
            f"代價是 {hz(H)} rollout 誤差由 {f3(c['free']['rollout'][-1])} 變為 {f3(c['constrained']['rollout'][-1])} °C。")
    if E and "holdout" in E:
        ho = E["holdout"]
        hT = sum(x["T_rmse"] for x in ho["holdout"]) / len(ho["holdout"])
        concl.append(f"插值能力：沒看過 14/18 °C 的模型單步 RMSE {f3(hT)}（看過時 {f3(ho['seen']['T_rmse'])}）。")
    if E:
        u = E["uncertainty"]
        concl.append(f"不確定度：長時間 rollout 的 ±2σ 覆蓋 {u['coverage_2sigma']:.0%} 的實際室溫"
                     + ("。" if u['coverage_2sigma'] >= 0.8 else "，集成低估誤差，只能作相對指標。"))

    # ─── 主表 ───
    rows = (f"<tr><td>Persistence（T[t+1] = T[t]）</td><td>{f4(per['T_rmse'])}</td><td>—</td>"
            f"<td>{f3(per['rollout'][k_mid - 1])}</td><td>{f3(rp)}</td></tr>")
    for name, s in [("MLP-only", mlp), ("CNN_MLP", cnn)]:
        rows += (f"<tr><td>{name}（{cfg['seeds']} seeds，mean ± std）</td><td>{pm(s['T_rmse'], f4)}</td><td>{pm(s['P_rmse'], f2)}</td>"
                 f"<td>{f3(s['rollout_mean'][k_mid - 1])} ± {f3(s['rollout_std'][k_mid - 1])}</td>"
                 f"<td>{f3(s['rollout_mean'][-1])} ± {f3(s['rollout_std'][-1])}</td></tr>")
    for name, e, hl in [("MLP-only 集成", me, ""), ("CNN_MLP 集成（交付的模擬器）", ce, ' class="hl"')]:
        rows += (f"<tr{hl}><td>{name}</td><td>{f4(e['T_rmse'])}</td><td>{f2(e['P_rmse'])}</td>"
                 f"<td>{f3(e['rollout'][k_mid - 1])}</td><td>{f3(e['rollout'][-1])}</td></tr>")
    main_tbl = (f"<table><tr><th>模型</th><th>單步室溫 RMSE (°C)</th><th>單步功率 RMSE (kW)</th>"
                f"<th>{hz(k_mid)} rollout (°C)</th><th>{hz(H)} rollout (°C)</th></tr>{rows}</table>")

    # ─── Day 1 資料檢查 ───
    data_html = "<p class='muted'>（未找到 data_check.json）</p>"
    if D:
        req = "".join(f"<tr><td>{k}</td><td>{'✅' if all(v.values()) else '❌'}</td><td>{', '.join(v)}</td></tr>"
                      for k, v in D["required"].items())
        dup = ""
        if D.get("duplicate_fault_files"):
            dup = "<li>內容完全相同的故障檔：" + "；".join(" = ".join(g) for g in D["duplicate_fault_files"]) + "</li>"
        p = D["processed"]
        data_html = f"""
<table><tr><th>Day 1 必要欄位</th><th>結果</th><th>欄位</th></tr>{req}</table>
<ul><li>原始資料：{D['start']} ～ {D['end']}，{D['rows']:,} 筆，每 {D['interval_min']:.0f} 分鐘，缺值 {D['missing']}。</li>
<li>控制量 CHWC_VLV_DM（營業時段）：平均 {D['valve']['mean']:.2f}、標準差 {D['valve']['std']:.2f}、開啟比例 {D['valve']['frac_open']:.0%}。</li>
<li>全年常數、無法使用的欄位：{', '.join(D['constant_cols'])}。</li>
<li>{D['units']['SA_SP_note']}；SA_CFM：{D['units']['SA_CFM_note']}。</li>
<li>{D['occupied']['note']}。</li>{dup}
<li>前處理後：{p['rows']:,} 筆（5 分鐘、只保留營業時段），平均功率 {p['power_kw_mean']:.1f} kW
（風扇 {p['fan_kw_mean']:.2f} kW + 冷卻負載 {p['cool_kw_mean']:.1f} kW ÷ COP {p['cop_assumed']:g}）。</li></ul>
<div class="note"><b>Day 1 決定：</b>{html.escape(D['decision'])}</div>
{img(fig, "fig0_data_overview.png", "圖 0　前處理後資料：室溫、控制量、功率（9 月一週）")}"""

    # ─── 進階評估 ───
    ev = ""
    if E:
        if "constraint" in E:
            c = E["constraint"]
            ev += f"""<h3>物理單調性約束</h3>
<p>原始資料由 PI 控制器產生：天氣熱時閥門開大、室溫也偏高。未加約束的模型會把這個「相關」學成「因果」，
在模擬中出現「閥門開越大越熱」。訓練時對「持續改變控制量」的梯度加入懲罰，強制
<b>閥門開大 → 室溫下降、功率上升</b>。這是閉環控制能成立的前提。</p>
<table><tr><th>版本</th><th>單步室溫</th><th>單步功率</th><th>{hz(H)} rollout</th><th>室溫方向違反率</th><th>功率方向違反率</th></tr>
<tr><td>無約束</td><td>{f4(c['free']['T_rmse'])}</td><td>{f2(c['free']['P_rmse'])}</td><td>{f3(c['free']['rollout'][-1])}</td>
<td>{c['free'].get('T_violation', 0):.1%}</td><td>{c['free'].get('P_violation', 0):.1%}</td></tr>
<tr class="hl"><td>有約束（交付版本）</td><td>{f4(c['constrained']['T_rmse'])}</td><td>{f2(c['constrained']['P_rmse'])}</td>
<td>{f3(c['constrained']['rollout'][-1])}</td><td>{c['constrained'].get('T_violation', 0):.1%}</td>
<td>{c['constrained'].get('P_violation', 0):.1%}</td></tr></table>
{img(fig, "fig9_constraint.png", "圖 9　物理約束消融：控制量響應與 rollout 誤差")}"""
        if "holdout" in E:
            ev += f"<h3>留出 setpoint</h3>{img(fig, 'fig5_holdout.png', '圖 5　留出 setpoint 的多步誤差')}"
        ev += f"""<h3>控制量響應掃描</h3>
<p>控制量固定在各個值做 rollout，畫出模型學到的「控制量 → 室溫／功率」關係。功率曲線就是 DE 最佳化面對的地形。</p>
{img(fig, "fig6_sweep.png", "圖 6　控制量對平均室溫與平均功率的響應")}
<h3>歷史視窗長度消融</h3>"""
        ab = E["ablation"]
        ev += ("<table><tr><th>視窗</th><th>單步室溫 (°C)</th><th>單步功率 (kW)</th><th>rollout (°C)</th><th>參數量</th><th>訓練秒數</th></tr>"
               + "".join(f"<tr><td>L = {k}（{spec.steps_to_h(int(k)):g} h）</td><td>{pm(v['T_rmse'], f4)}</td><td>{pm(v['P_rmse'], f2)}</td>"
                         f"<td>{pm(v['roll_end'])}</td><td>{int(v['params']['mean']):,}</td><td>{v['seconds']['mean']:.0f}</td></tr>"
                         for k, v in ab.items()) + "</table>")
        ev += img(fig, "fig7_ablation.png", "圖 7　歷史視窗長度消融")
        ev += f"<h3>集成不確定度</h3>{img(fig, 'fig8_uncertainty.png', '圖 8　長時間自我回饋 rollout 與 ±2σ')}"

    notes = "".join(f"<li>{html.escape(n)}</li>" for n in spec.notes)
    body = f"""
<header><h1>HVAC 模擬器專題 — CNN 動態模型報告</h1>
<p>{html.escape(spec.title)}　｜　產生時間 {datetime.now():%Y-%m-%d %H:%M}　｜　訓練 {cfg['n_train']:,} / 驗證 {cfg['n_val']:,} / 測試 {cfg['n_test']:,} 筆
　｜　視窗 {L} 步（{spec.steps_to_h(L):g} h）　｜　{cfg['seeds']} seeds</p></header>
<nav><button class="on">總覽</button><button>Day 1 資料</button><button>方法</button><button>Day 2 主實驗</button><button>進階評估</button><button>限制</button></nav>
<main>
<section class="on"><h2>總覽</h2>{cards}<h3>結論</h3><ul>{''.join(f'<li>{c}</li>' for c in concl)}</ul></section>
<section><h2>Day 1　資料檢查與前處理</h2>{data_html}</section>
<section><h2>方法</h2>
<p>CNN 是閉環中的<b>受控體模擬器</b>：控制器每 {spec.dt_min} 分鐘給出控制量（{html.escape(spec.act_label)}），
模型預測下一步的室溫變化量 ΔT 與 HVAC 功率 P，DE 再依模擬結果最佳化模糊參數。</p>
{arch_svg(spec, L)}
<ul><li>時間切分：訓練 {cfg['train_months']} 月、驗證 {cfg['val_months']} 月、測試 {cfg['test_months']} 月（不隨機切，避免相鄰時間點洩漏）；標準化只用訓練集。</li>
<li>預測 ΔT 而非絕對溫度；歷史視窗只放室溫、控制量與外生變數，避免 rollout 時偷看受控制影響的真值。</li>
<li>損失：標準化 ΔT 與 P 的加權 MSE（1 : 0.5）＋ 物理單調性懲罰（λ = {cfg['mono']:g}）；Adam、ReduceLROnPlateau、early stopping。</li>
<li>對照組：MLP-only（相同輸入直接攤平）、persistence（T[t+1] = T[t]）。</li>
<li>多步 rollout：只給初始 {spec.steps_to_h(L):g} h 真值，之後室溫用模型預測回填，控制量與外生變數用實際紀錄。</li>
<li>交付的模擬器 = {cfg['seeds']} 個 seed 的集成平均：<code>HVACSimulator.load("out/{spec.name}/models")</code>。</li></ul></section>
<section><h2>Day 2　主實驗（測試集 = {cfg['test_months']} 月）</h2>{main_tbl}
{img(fig, "fig2_pred_vs_actual.png", "圖 2　★ 預測 vs 實際：室溫、誤差、功率")}
{img(fig, "fig4_rollout.png", "圖 4　★ 多步 rollout 室溫誤差")}
{img(fig, "fig3_scatter.png", "圖 3　預測 vs 實際散佈圖（顏色 = 控制量）")}
{img(fig, "fig1_loss.png", "圖 1　訓練與驗證 loss（每條一個 seed）")}</section>
<section><h2>進階評估</h2>{ev or "<p class='muted'>尚未執行 scripts/evaluate_cnn.py</p>"}</section>
<section><h2>限制與未來工作</h2><ul>{notes}
<li>未來工作：以 EnergyPlus / Modelica 產生控制量隨機激勵的資料，降低閉迴路混淆；比較 LSTM / TCN；以集成不確定度做 robust 最佳化。</li></ul></section>
</main>"""
    page = (f'<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>CNN 動態模型報告</title>'
            f'<style>{CSS}</style></head><body>{body}<script>{JS}</script></body></html>')
    dst = out / "cnn_report.html"
    dst.write_text(page, encoding="utf-8")
    print("報告 →", dst.resolve())


if __name__ == "__main__":
    main()
