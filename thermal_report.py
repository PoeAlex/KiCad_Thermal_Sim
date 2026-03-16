"""
HTML report generation for ThermalSim.

This module generates a compact, printable HTML report summarizing the
thermal simulation results, including static images and an interactive
heatmap viewer for the final temperature field.
"""

import html
import json
import os
import re


REPORT_STYLE = """
<style>
  :root{color-scheme:light;--bg:#f4f6f8;--panel:#fff;--alt:#f8fafc;--ink:#17202a;--muted:#5f6b76;--line:#d7dee5;--strong:#b9c3ce;--accent:#1f5f8b;--accent-soft:#e8f1f7;--danger:#9b2c2c;--shadow:0 10px 28px rgba(23,32,42,.08);--radius:14px;--mono:"Cascadia Mono","SFMono-Regular",Consolas,"Liberation Mono",monospace;--sans:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif}
  *{box-sizing:border-box} body{margin:0;font-family:var(--sans);background:linear-gradient(180deg,#eef3f6 0%,var(--bg) 220px);color:var(--ink);line-height:1.5}
  .report-shell{max-width:1380px;margin:0 auto;padding:24px}.report-header{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:22px}.report-title{margin:0;font-size:2rem;line-height:1.1}.report-subtitle{margin:8px 0 0;color:var(--muted);max-width:760px}.report-meta{color:var(--muted);font-size:.94rem;text-align:right;white-space:nowrap}
  .section-card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);padding:18px 20px;margin-bottom:18px;break-inside:avoid}.section-card.compact{padding:14px 16px}.section-title{margin:0 0 12px;font-size:1.15rem;display:flex;align-items:center;justify-content:space-between;gap:10px}.section-note,.small{color:var(--muted);font-size:.92rem}
  .summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px}.metric-card{border:1px solid var(--line);border-radius:12px;background:linear-gradient(180deg,#fff 0%,var(--alt) 100%);padding:12px 14px;min-height:88px}.metric-label{margin:0 0 6px;font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}.metric-value{margin:0;font-size:1.2rem;font-weight:700}.metric-detail{margin:6px 0 0;font-size:.85rem;color:var(--muted)}
  .results-grid,.details-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.image-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.image-card{border:1px solid var(--line);border-radius:12px;padding:12px;background:#fff}.image-card h3{margin:0 0 10px;font-size:1rem}.image-card img{width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fff;display:block}
  .table-wrap{overflow-x:auto} table{width:100%;border-collapse:collapse;margin:0;font-size:.95rem} th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top} th{background:#eef3f7;font-weight:700} tbody tr:nth-child(even) td{background:#fbfcfd}.mono{font-family:var(--mono);font-size:.88rem}
  details{border:1px solid var(--line);border-radius:12px;background:#fff;margin-top:12px;overflow:hidden} details summary{cursor:pointer;list-style:none;padding:12px 14px;font-weight:700;background:#f5f8fb;border-bottom:1px solid transparent} details[open] summary{border-bottom-color:var(--line)} details summary::-webkit-details-marker{display:none}.details-body{padding:14px}
  pre{margin:0;padding:12px;background:#0f1720;color:#dde7ef;border-radius:10px;overflow-x:auto;font-size:.83rem;white-space:pre-wrap;word-break:break-word}.spacer{height:16px}.spacer-sm{height:12px}
  .viewer-shell{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(320px,.95fr);gap:18px;align-items:start}.viewer-panel{border:1px solid var(--line);border-radius:12px;padding:14px;background:linear-gradient(180deg,#fff 0%,#f9fbfd 100%)}.viewer-controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:10px}.viewer-controls label{font-size:.92rem;color:var(--muted)}.viewer-controls select,.viewer-controls button{font:inherit;border:1px solid var(--strong);border-radius:9px;background:#fff;color:var(--ink);padding:7px 10px}.viewer-controls button{cursor:pointer}.viewer-controls button:hover{background:var(--accent-soft);border-color:var(--accent)}
  .viewer-help{margin:0 0 12px;color:var(--muted);font-size:.92rem}.legend-row{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:12px}.legend-bar{flex:1;height:12px;border-radius:999px;border:1px solid var(--line);background:linear-gradient(90deg,#0b0c32 0%,#420a68 18%,#781c6d 36%,#bb3754 56%,#ec6824 76%,#f6d645 100%)}.legend-labels{display:flex;justify-content:space-between;gap:12px;color:var(--muted);font-size:.84rem;min-width:140px}
  .viewer-canvas-wrap{position:relative;border-radius:12px;overflow:hidden;border:1px solid var(--line);background:linear-gradient(45deg,#eef2f6 25%,transparent 25%),linear-gradient(-45deg,#eef2f6 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#eef2f6 75%),linear-gradient(-45deg,transparent 75%,#eef2f6 75%);background-size:20px 20px;background-position:0 0,0 10px,10px -10px,-10px 0;min-height:260px}.viewer-canvas-wrap canvas{display:block;width:100%;height:auto;image-rendering:pixelated}#roi-overlay{position:absolute;inset:0;pointer-events:none}
  .viewer-tooltip{position:fixed;z-index:50;pointer-events:none;min-width:170px;padding:8px 10px;border-radius:10px;background:rgba(15,23,32,.92);color:#f3f7fa;box-shadow:0 8px 26px rgba(0,0,0,.25);font-size:.84rem;display:none;white-space:nowrap}.roi-table-caption{margin:0 0 10px;color:var(--muted);font-size:.92rem}.delete-btn{border:1px solid #d9a8a8;color:var(--danger);background:#fff6f6;border-radius:8px;padding:4px 8px;cursor:pointer;font-size:.83rem}.delete-btn:hover{background:#ffe7e7}
  .snapshot-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}.snapshot-card{border:1px solid var(--line);border-radius:12px;padding:10px;background:#fff}.snapshot-card img{width:100%;height:auto;display:block;border-radius:10px;border:1px solid var(--line)}.snapshot-card figcaption{color:var(--muted);font-size:.88rem;margin-top:8px}.empty-state{border:1px dashed var(--strong);border-radius:12px;padding:16px;color:var(--muted);background:#fafcfd}
  @media (max-width:1080px){.results-grid,.details-grid,.viewer-shell,.image-grid{grid-template-columns:1fr}.report-header{flex-direction:column}.report-meta{text-align:left;white-space:normal}}
  @media print{:root{--bg:#fff;--panel:#fff;--alt:#fff;--ink:#000;--muted:#444;--line:#bdbdbd;--strong:#969696;--shadow:none}body{background:#fff;font-size:11pt}.report-shell{max-width:none;padding:0}.section-card,.image-card,.viewer-panel,.snapshot-card,details{box-shadow:none;break-inside:avoid}.viewer-controls,.viewer-help,.viewer-tooltip,.delete-btn{display:none!important}.viewer-shell,.results-grid,.details-grid,.image-grid,.summary-grid{grid-template-columns:1fr}a{color:inherit;text-decoration:none}}
</style>
"""


REPORT_SCRIPT = """
<script>
(()=>{function c(v,min,max){return Math.max(min,Math.min(max,v));}function l(a,b,t){return a+(b-a)*t;}function h(hex){hex=hex.replace('#','');return{r:parseInt(hex.slice(0,2),16),g:parseInt(hex.slice(2,4),16),b:parseInt(hex.slice(4,6),16)};}function color(val,vmin,vmax){if(val===null||val===undefined||!isFinite(val))return{r:220,g:226,b:232,a:255};const stops=[{t:0,c:h('#0b0c32')},{t:.18,c:h('#420a68')},{t:.36,c:h('#781c6d')},{t:.56,c:h('#bb3754')},{t:.76,c:h('#ec6824')},{t:1,c:h('#f6d645')}];const norm=vmax<=vmin?0:c((val-vmin)/(vmax-vmin),0,1);for(let i=1;i<stops.length;i++){if(norm<=stops[i].t){const p=stops[i-1],n=stops[i],t=(norm-p.t)/Math.max(n.t-p.t,1e-9);return{r:Math.round(l(p.c.r,n.c.r,t)),g:Math.round(l(p.c.g,n.c.g,t)),b:Math.round(l(p.c.b,n.c.b,t)),a:255};}}const end=stops[stops.length-1].c;return{r:end.r,g:end.g,b:end.b,a:255};}function fmt(v,d){if(v===null||v===undefined||!isFinite(v))return'n/a';return Number(v).toFixed(d);}function rect(a,b,layer){if(!a||!b||!layer)return null;return{col0:c(Math.min(a.col,b.col),0,layer.cols-1),col1:c(Math.max(a.col,b.col),0,layer.cols-1),row0:c(Math.min(a.row,b.row),0,layer.rows-1),row1:c(Math.max(a.row,b.row),0,layer.rows-1)};}
document.addEventListener('DOMContentLoaded',()=>{const node=document.getElementById('interactive-heatmap-data');if(!node)return;let payload=null;try{payload=JSON.parse(node.textContent);}catch(err){console.error('Failed to parse interactive heatmap payload',err);return;}if(!payload||!payload.layers||!payload.layers.length)return;
const sel=document.getElementById('heatmap-layer-select'),canvas=document.getElementById('heatmap-canvas'),overlay=document.getElementById('roi-overlay'),tip=document.getElementById('heatmap-tooltip'),tbody=document.getElementById('roi-table-body'),empty=document.getElementById('roi-empty-state'),clearCurrent=document.getElementById('clear-current-rois'),clearAll=document.getElementById('clear-all-rois'),active=document.getElementById('active-layer-name'),legendMin=document.getElementById('legend-min'),legendMax=document.getElementById('legend-max');
const state={rois:[],dragStart:null,dragCurrent:null,dragLayerIndex:null,nextRoiId:1,currentLayerIndex:payload.layers[0].index};const ctx=canvas.getContext('2d'),octx=overlay.getContext('2d');
function getLayer(idx){return payload.layers.find(layer=>layer.index===idx)||payload.layers[0];}function currentLayer(){return getLayer(state.currentLayerIndex);}function setSizes(layer){canvas.width=layer.cols;canvas.height=layer.rows;overlay.width=layer.cols;overlay.height=layer.rows;}function cellValue(layer,row,col){return layer.data[row*layer.cols+col];}
function buildOptions(){sel.innerHTML='';payload.layers.forEach(layer=>{const opt=document.createElement('option');opt.value=String(layer.index);opt.textContent=layer.name;if(layer.index===state.currentLayerIndex)opt.selected=true;sel.appendChild(opt);});}
function drawHeatmap(){const layer=currentLayer();setSizes(layer);const img=ctx.createImageData(layer.cols,layer.rows);for(let i=0;i<layer.data.length;i++){const rgba=color(layer.data[i],payload.vmin_c,payload.vmax_c),off=i*4;img.data[off]=rgba.r;img.data[off+1]=rgba.g;img.data[off+2]=rgba.b;img.data[off+3]=rgba.a;}ctx.putImageData(img,0,0);active.textContent=layer.name;legendMin.textContent=fmt(payload.vmin_c,1)+' C';legendMax.textContent=fmt(payload.vmax_c,1)+' C';drawOverlay();}
function canvasCell(evt){const layer=currentLayer(),r=canvas.getBoundingClientRect();if(!r.width||!r.height)return null;const x=c(evt.clientX-r.left,0,r.width-.0001),y=c(evt.clientY-r.top,0,r.height-.0001);return{col:c(Math.floor(x/r.width*layer.cols),0,layer.cols-1),row:c(Math.floor(y/r.height*layer.rows),0,layer.rows-1)};}
function showTip(evt,cell){const layer=currentLayer();if(!cell){tip.style.display='none';return;}const value=cellValue(layer,cell.row,cell.col),xMm=payload.x_min_mm+cell.col*payload.res_mm,yMm=payload.y_min_mm+cell.row*payload.res_mm;tip.innerHTML='<strong>'+layer.name+'</strong><br>Temp: '+fmt(value,2)+' C<br>Row/Col: '+cell.row+' / '+cell.col+'<br>X/Y: '+fmt(xMm,3)+' / '+fmt(yMm,3)+' mm';tip.style.left=(evt.clientX+16)+'px';tip.style.top=(evt.clientY+16)+'px';tip.style.display='block';}
function drawRect(r,stroke,fill,dashed){const w=r.col1-r.col0+1,h=r.row1-r.row0+1;octx.save();octx.strokeStyle=stroke;octx.fillStyle=fill;octx.lineWidth=1.25;octx.setLineDash(dashed?[3,2]:[]);octx.fillRect(r.col0,r.row0,w,h);octx.strokeRect(r.col0+.5,r.row0+.5,Math.max(w-1,1),Math.max(h-1,1));octx.restore();}
function drawOverlay(){const layer=currentLayer();octx.clearRect(0,0,overlay.width,overlay.height);state.rois.forEach(roi=>{if(roi.layerIndex!==layer.index)return;drawRect(roi,'rgba(12,92,152,.95)','rgba(31,95,139,.18)',false);});if(state.dragStart&&state.dragCurrent&&state.dragLayerIndex===layer.index){const draft=rect(state.dragStart,state.dragCurrent,layer);if(draft)drawRect(draft,'rgba(155,44,44,.95)','rgba(155,44,44,.16)',true);}}
function stats(layer,roi){let min=Infinity,max=-Infinity,sum=0,count=0;for(let row=roi.row0;row<=roi.row1;row++){for(let col=roi.col0;col<=roi.col1;col++){const value=cellValue(layer,row,col);if(value===null||value===undefined||!isFinite(value))continue;min=Math.min(min,value);max=Math.max(max,value);sum+=value;count++;}}return{min:count?min:null,max:count?max:null,avg:count?sum/count:null,cells:count};}
function renderTable(){tbody.innerHTML='';if(!state.rois.length){empty.style.display='block';return;}empty.style.display='none';state.rois.forEach((roi,index)=>{const layer=getLayer(roi.layerIndex),s=stats(layer,roi),row=document.createElement('tr');row.innerHTML='<td>ROI '+(index+1)+'</td><td>'+layer.name+'</td><td>'+roi.col0+', '+roi.row0+'</td><td>'+roi.col1+', '+roi.row1+'</td><td>'+(roi.col1-roi.col0+1)+' x '+(roi.row1-roi.row0+1)+'</td><td>'+s.cells+'</td><td>'+fmt(s.min,2)+' C</td><td>'+fmt(s.max,2)+' C</td><td>'+fmt(s.avg,2)+' C</td><td><button class=\"delete-btn\" data-roi-id=\"'+roi.id+'\">Delete</button></td>';tbody.appendChild(row);});}
function finishDrag(){if(!state.dragStart||!state.dragCurrent){state.dragStart=null;state.dragCurrent=null;state.dragLayerIndex=null;drawOverlay();return;}const layer=getLayer(state.dragLayerIndex),r=rect(state.dragStart,state.dragCurrent,layer);if(r){r.id=state.nextRoiId++;r.layerIndex=layer.index;state.rois.push(r);renderTable();}state.dragStart=null;state.dragCurrent=null;state.dragLayerIndex=null;drawOverlay();}
sel.addEventListener('change',()=>{state.currentLayerIndex=parseInt(sel.value,10);drawHeatmap();});canvas.addEventListener('mousedown',evt=>{const cell=canvasCell(evt);state.dragStart=cell;state.dragCurrent=cell;state.dragLayerIndex=state.currentLayerIndex;drawOverlay();});canvas.addEventListener('mousemove',evt=>{const cell=canvasCell(evt);showTip(evt,cell);if(!state.dragStart)return;state.dragCurrent=cell;drawOverlay();});canvas.addEventListener('mouseleave',()=>{tip.style.display='none';});window.addEventListener('mouseup',()=>{if(state.dragStart)finishDrag();});tbody.addEventListener('click',evt=>{if(!evt.target.classList.contains('delete-btn'))return;const roiId=parseInt(evt.target.getAttribute('data-roi-id'),10);state.rois=state.rois.filter(roi=>roi.id!==roiId);renderTable();drawOverlay();});clearCurrent.addEventListener('click',()=>{state.rois=state.rois.filter(roi=>roi.layerIndex!==state.currentLayerIndex);renderTable();drawOverlay();});clearAll.addEventListener('click',()=>{state.rois=[];renderTable();drawOverlay();});buildOptions();drawHeatmap();renderTable();});})();
</script>
"""


def _fmt(val, suffix=""):
    """Format a value for display in the report."""
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return f"{val:.4f}{suffix}"
    return f"{val}{suffix}"


def _esc(text):
    """HTML-escape a text value."""
    return html.escape(text if text is not None else "")


def _json_for_script(data):
    """Serialize JSON for safe embedding in a script tag."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")


def _table_html(headers, rows, empty_text="n/a", table_class=""):
    """Render a simple HTML table."""
    class_attr = f" class='{_esc(table_class)}'" if table_class else ""
    thead = "".join(f"<th>{_esc(str(header))}</th>" for header in headers)
    if rows:
        body = "\n".join(
            "<tr>" + "".join(f"<td>{_esc(str(cell))}</td>" for cell in row) + "</tr>"
            for row in rows
        )
    else:
        body = f"<tr><td colspan='{len(headers)}'>{_esc(empty_text)}</td></tr>"
    return (
        f"<div class='table-wrap'><table{class_attr}>"
        f"<thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _mapping_table_html(mapping, empty_text="n/a", key_label="Key", value_label="Value"):
    """Render a mapping as a two-column HTML table."""
    rows = [(str(key), _fmt(value)) for key, value in mapping.items()]
    return _table_html([key_label, value_label], rows, empty_text=empty_text)


def _details_block(title, body_html, open_by_default=False):
    """Render a collapsible details block."""
    open_attr = " open" if open_by_default else ""
    return (
        f"<details{open_attr}>"
        f"<summary>{_esc(title)}</summary>"
        f"<div class='details-body'>{body_html}</div>"
        f"</details>"
    )


def _build_snapshot_items(out_dir, snapshot_files):
    """Normalize snapshot metadata."""
    snapshot_items = []
    if snapshot_files is not None:
        for item in snapshot_files:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                t_val, fname = item
            else:
                fname = os.path.basename(str(item))
                match = re.search(r"_t([0-9.]+)", fname)
                t_val = float(match.group(1)) if match else None
            snapshot_items.append((t_val, os.path.basename(fname)))
    else:
        try:
            import glob
            for path in glob.glob(os.path.join(out_dir, "snap_*.png")):
                fname = os.path.basename(path)
                match = re.search(r"_t([0-9.]+)", fname)
                t_val = float(match.group(1)) if match else None
                snapshot_items.append((t_val, fname))
        except Exception:
            snapshot_items = []
    snapshot_items.sort(key=lambda item: (item[0] if item[0] is not None else 1e9, item[1]))
    return snapshot_items


def _build_summary_metrics(settings, interactive_heatmap, k_norm_info, snapshot_debug):
    """Build top-level summary metrics."""
    layers = (interactive_heatmap or {}).get("layers", [])
    top_layer = layers[0] if layers else None
    bottom_layer = layers[-1] if layers else None
    layer_max_values = [layer.get("max_c") for layer in layers if layer.get("max_c") is not None]
    overall_peak = max(layer_max_values) if layer_max_values else None
    grid_label = "n/a"
    if layers:
        grid_label = f"{layers[0].get('rows', 0)} x {layers[0].get('cols', 0)}"
    metrics = [
        ("Duration", f"{float(settings.get('time', 0.0)):.2f} s" if "time" in settings else "n/a", None),
        ("Ambient", f"{float(settings.get('amb', 0.0)):.2f} C" if "amb" in settings else "n/a", None),
        ("Resolution", f"{float(settings.get('res', 0.0)):.3f} mm" if "res" in settings else "n/a", grid_label),
        ("Visible Layers", str(len(layers)), ", ".join(layer.get("name", "") for layer in layers) if layers else None),
        ("Top Peak", f"{float(top_layer.get('max_c')):.2f} C" if top_layer and top_layer.get("max_c") is not None else "n/a", top_layer.get("name") if top_layer else None),
        ("Bottom Peak", f"{float(bottom_layer.get('max_c')):.2f} C" if bottom_layer and bottom_layer.get("max_c") is not None else "n/a", bottom_layer.get("name") if bottom_layer else None),
        ("Overall Peak", f"{float(overall_peak):.2f} C" if overall_peak is not None else "n/a", None),
        ("Solver", str((k_norm_info or {}).get("backend", "n/a")), f"steps: {(k_norm_info or {}).get('steps_total', 'n/a')}"),
        ("Input Power", f"{float((k_norm_info or {}).get('pin_w')):.3f} W" if (k_norm_info or {}).get("pin_w") is not None else "n/a", None),
        ("Final Cooling", f"{float((k_norm_info or {}).get('pout_final_w')):.3f} W" if (k_norm_info or {}).get("pout_final_w") is not None else "n/a", None),
        ("Snapshots", "enabled" if settings.get("snapshots") else "disabled", f"target: {settings.get('snap_count', 0)}"),
        ("Output Folder", os.path.basename(str((snapshot_debug or {}).get("run_dir", ""))) or "n/a", None),
    ]
    cards = []
    for label, value, detail in metrics:
        detail_html = f"<p class='metric-detail'>{_esc(detail)}</p>" if detail else ""
        cards.append(
            "<article class='metric-card'>"
            f"<p class='metric-label'>{_esc(label)}</p>"
            f"<p class='metric-value'>{_esc(value)}</p>"
            f"{detail_html}"
            "</article>"
        )
    return "<div class='summary-grid'>" + "".join(cards) + "</div>"


def _build_image_card(title, path, empty_message):
    """Render an image card for a static result image."""
    if path:
        src = _esc(os.path.basename(path))
        content = f"<img src='{src}' alt='{_esc(title)}'>"
    else:
        content = f"<div class='empty-state'>{_esc(empty_message)}</div>"
    return "<article class='image-card'>" f"<h3>{_esc(title)}</h3>{content}</article>"


def _build_interactive_viewer(interactive_heatmap):
    """Render the interactive heatmap viewer block."""
    if not interactive_heatmap or not interactive_heatmap.get("layers"):
        return "<div class='empty-state'>Interactive heatmap data is not available for this run.</div>"
    payload_json = _json_for_script(interactive_heatmap)
    return (
        "<div class='viewer-shell'>"
        "<div class='viewer-panel'>"
        "<div class='viewer-controls'>"
        "<label for='heatmap-layer-select'>Layer</label>"
        "<select id='heatmap-layer-select' aria-label='Active heatmap layer'></select>"
        "<button id='clear-current-rois' type='button'>Clear current layer</button>"
        "<button id='clear-all-rois' type='button'>Clear all</button>"
        "</div>"
        "<p class='viewer-help'>Hover to inspect cell temperature. Drag to create ROI rectangles.</p>"
        "<div class='legend-row'><div class='legend-bar' aria-hidden='true'></div>"
        "<div class='legend-labels'><span id='legend-min'>n/a</span><span id='legend-max'>n/a</span></div></div>"
        "<div class='viewer-canvas-wrap'><canvas id='heatmap-canvas'></canvas><canvas id='roi-overlay'></canvas></div>"
        "<p class='small'>Active layer: <strong id='active-layer-name'>n/a</strong></p>"
        "</div>"
        "<div class='viewer-panel'>"
        "<h3 class='section-title'>ROI Statistics</h3>"
        "<p class='roi-table-caption'>Each ROI uses raw cell values from the layer where it was drawn.</p>"
        "<div id='roi-empty-state' class='empty-state'>No ROI selected yet.</div>"
        "<div class='table-wrap'><table id='roi-table'><thead><tr>"
        "<th>ROI</th><th>Layer</th><th>Start</th><th>End</th><th>Size</th><th>Cells</th>"
        "<th>Min</th><th>Max</th><th>Avg</th><th>Action</th>"
        "</tr></thead><tbody id='roi-table-body'></tbody></table></div>"
        "</div></div>"
        "<div id='heatmap-tooltip' class='viewer-tooltip'></div>"
        f"<script id='interactive-heatmap-data' type='application/json'>{payload_json}</script>"
    )


def _build_snapshot_gallery(snapshot_items):
    """Render snapshot gallery HTML."""
    if not snapshot_items:
        return "<div class='empty-state'>No snapshots captured.</div>"
    cards = []
    for t_val, fname in snapshot_items:
        label = f"t = {t_val:.1f} s" if t_val is not None else fname
        cards.append(
            "<figure class='snapshot-card'>"
            f"<img src='{_esc(fname)}' alt='{_esc(label)}'>"
            f"<figcaption>{_esc(label)}</figcaption>"
            "</figure>"
        )
    return "<div class='snapshot-grid'>" + "".join(cards) + "</div>"


def write_html_report(
    settings,
    stack_info,
    stackup_derived,
    pad_power,
    layer_names,
    preview_path,
    heatmap_path,
    k_norm_info=None,
    out_dir=None,
    snapshot_debug=None,
    snapshot_files=None,
    interactive_heatmap=None
):
    """Generate an HTML report for the thermal simulation."""
    out_dir = out_dir or os.path.dirname(__file__)
    report_path = os.path.join(out_dir, "thermal_report.html")
    settings = settings or {}
    stack_info = stack_info or {}
    stackup_derived = stackup_derived or {}
    k_norm_info = k_norm_info or {}
    snapshot_debug = snapshot_debug or {}
    interactive_heatmap = interactive_heatmap or {}

    total_thick_mm = stackup_derived.get("total_thick_mm_used")
    board_thick_mm = stackup_derived.get("stack_board_thick_mm")
    copper_thicknesses = stackup_derived.get("copper_thickness_mm_used", [])
    gaps_used = stackup_derived.get("gap_mm_used", [])
    gap_fallback_used = stackup_derived.get("gap_fallback_used", False)
    snapshot_items = _build_snapshot_items(out_dir, snapshot_files)
    preview_rel = os.path.basename(preview_path) if preview_path else ""
    heatmap_rel = os.path.basename(heatmap_path) if heatmap_path else ""

    copper_rows = []
    for i, thickness in enumerate(copper_thicknesses):
        copper_rows.append((layer_names[i] if i < len(layer_names) else f"Layer {i}", _fmt(thickness, " mm")))
    gap_rows = []
    for i, gap in enumerate(gaps_used):
        src = layer_names[i] if i < len(layer_names) else f"Layer {i}"
        dst = layer_names[i + 1] if (i + 1) < len(layer_names) else f"Layer {i + 1}"
        gap_rows.append((f"{src} -> {dst}", _fmt(gap, " mm")))
    fr4_eff_rows = []
    for i, value in enumerate(k_norm_info.get("t_fr4_eff_per_plane_mm") or []):
        fr4_eff_rows.append((layer_names[i] if i < len(layer_names) else f"Layer {i}", _fmt(value, " mm")))

    settings_rows = [(str(key), str(value)) for key, value in settings.items()]
    pad_rows = [(str(name), _fmt(power, " W")) for name, power in pad_power]
    thickness_summary_rows = [
        ("Board thickness (stackup)", _fmt(board_thick_mm, " mm")),
        ("Total thickness used", _fmt(total_thick_mm, " mm")),
        ("Uniform gap fallback used", str(bool(gap_fallback_used))),
        ("Layer names", ", ".join(layer_names) if layer_names else "n/a"),
    ]

    solver_summary = {
        "strategy": k_norm_info.get("strategy"),
        "backend": k_norm_info.get("backend"),
        "multi_phase": k_norm_info.get("multi_phase"),
        "N": k_norm_info.get("N"),
        "nnz_K": k_norm_info.get("nnz_K"),
        "steps_total": k_norm_info.get("steps_total"),
        "factorizations": k_norm_info.get("factorizations"),
        "avg_solve_s": k_norm_info.get("avg_solve_s"),
        "pin_w": k_norm_info.get("pin_w"),
        "pout_final_w": k_norm_info.get("pout_final_w"),
        "steady_rel_diff": k_norm_info.get("steady_rel_diff"),
    }
    solver_summary = {key: value for key, value in solver_summary.items() if value is not None}

    images_html = "<div class='image-grid'>" \
        f"{_build_image_card('Geometry Preview', preview_rel, 'Preview image not available.')}" \
        f"{_build_image_card('Final Heatmap', heatmap_rel, 'Heatmap image not available.')}" \
        "</div>"

    results_html = (
        "<section class='section-card'>"
        "<h2 class='section-title'>Results</h2>"
        "<p class='section-note'>Static images stay embedded, and the final heatmap below is interactive.</p>"
        f"{images_html}<div class='spacer'></div>{_build_interactive_viewer(interactive_heatmap)}"
        "</section>"
    )

    details_html = (
        "<section class='section-card'>"
        "<h2 class='section-title'>Details</h2>"
        "<div class='details-grid'><div>"
        "<h3 class='section-title'>Stackup</h3>"
        f"{_table_html(['Metric', 'Value'], thickness_summary_rows)}<div class='spacer-sm'></div>"
        f"{_table_html(['Layer', 'Thickness'], copper_rows)}<div class='spacer-sm'></div>"
        f"{_table_html(['Interface', 'Gap'], gap_rows)}<div class='spacer-sm'></div>"
        "<h3 class='section-title'>Effective Dielectric Thickness</h3>"
        f"{_table_html(['Plane', 't_fr4_eff'], fr4_eff_rows)}"
        "</div><div>"
        "<h3 class='section-title'>Simulation Inputs</h3>"
        f"{_table_html(['Setting', 'Value'], settings_rows, empty_text='No settings recorded.')}<div class='spacer-sm'></div>"
        "<h3 class='section-title'>Power per Pad</h3>"
        f"{_table_html(['Pad', 'Power'], pad_rows, empty_text='No pad power recorded.')}"
        "</div></div>"
        f"{_details_block('Snapshots', _build_snapshot_gallery(snapshot_items), open_by_default=False)}"
        "</section>"
    )

    diagnostics_html = (
        "<section class='section-card compact'>"
        "<h2 class='section-title'>Diagnostics</h2>"
        "<p class='section-note'>Debug data is kept for troubleshooting, but collapsed by default.</p>"
        f"{_details_block('Solver Summary', _mapping_table_html(solver_summary), open_by_default=False)}"
        f"{_details_block('Solver Normalization and Debug', _mapping_table_html(k_norm_info), open_by_default=False)}"
        f"{_details_block('Snapshot Debug', _mapping_table_html(snapshot_debug), open_by_default=False)}"
        f"{_details_block('Raw Solver JSON', '<pre>' + _esc(json.dumps(k_norm_info, indent=2, sort_keys=True)) + '</pre>', open_by_default=False)}"
        f"{_details_block('Raw Snapshot JSON', '<pre>' + _esc(json.dumps(snapshot_debug, indent=2, sort_keys=True)) + '</pre>', open_by_default=False)}"
        "</section>"
    )

    html_body = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>KiCad Thermal Sim Report</title>"
        f"{REPORT_STYLE}</head><body><div class='report-shell'>"
        "<header class='report-header'><div><h1 class='report-title'>KiCad Thermal Sim Report</h1>"
        "<p class='report-subtitle'>Compact summary, print-friendly layout, static deliverables, and an interactive final heatmap.</p>"
        "</div><div class='report-meta'>"
        f"<div>Layers: {_esc(', '.join(layer_names) if layer_names else 'n/a')}</div>"
        f"<div>Report file: {_esc(os.path.basename(report_path))}</div>"
        "</div></header>"
        "<section class='section-card'><h2 class='section-title'>Summary</h2>"
        f"{_build_summary_metrics(settings, interactive_heatmap, k_norm_info, snapshot_debug)}</section>"
        f"{results_html}{details_html}{diagnostics_html}</div>{REPORT_SCRIPT}</body></html>"
    )

    try:
        with open(report_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(html_body)
    except Exception:
        return None
    return report_path
