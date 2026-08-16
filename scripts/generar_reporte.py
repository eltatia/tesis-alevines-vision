"""
generar_reporte.py
------------------
Genera un REPORTE PDF completo de la tesis (metodologia + metricas + graficos)
a partir de los CSV de resultados. Multipagina con matplotlib (PdfPages).

Salida: reports/REPORTE_TESIS.pdf

Uso:  python scripts/generar_reporte.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ---------- estilo ----------
OKABE = ['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#56B4E9','#8C6BB1']
TEAL='#12726b'; INK='#1a2b2a'; SOFT='#4a5d5c'; FAINT='#8aa'; GRID='#dfe6e6'
plt.rcParams.update({
 'font.family':'DejaVu Sans','font.size':10,'axes.edgecolor':'#c7d2d2',
 'axes.labelcolor':INK,'text.color':INK,'xtick.color':SOFT,'ytick.color':SOFT,
 'axes.grid':True,'grid.color':GRID,'grid.linewidth':.8,'axes.axisbelow':True,
 'axes.spines.top':False,'axes.spines.right':False,
})
A4=(8.27,11.69)
R=Path('reports'); C=Path('data/counts')

def load(p):
    try: return pd.read_csv(p)
    except Exception: return None

def header(fig,num,title):
    fig.text(.08,.945,num,fontsize=13,color='white',weight='bold',
             bbox=dict(boxstyle='round,pad=.5',fc=TEAL,ec='none'))
    fig.text(.15,.947,title,fontsize=17,color=INK,weight='bold',va='center')
    fig.add_artist(plt.Line2D([.08,.92],[.925,.925],color=TEAL,lw=1.5,transform=fig.transFigure))

def footer(fig,n):
    fig.text(.92,.03,f'{n}',ha='right',fontsize=8,color=FAINT)
    fig.text(.08,.03,'Conteo automatico de alevines · CITE Madre de Dios · Tesis 2026',
             fontsize=8,color=FAINT)

def bars(ax,cats,vals,color,fmt='{:.3f}',ymax=None,rot=0):
    x=np.arange(len(cats)); b=ax.bar(x,vals,color=color,width=.6,zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(cats,rotation=rot,ha='center' if rot==0 else 'right',fontsize=8.5)
    for xi,v in zip(x,vals):
        ax.text(xi,v,fmt.format(v),ha='center',va='bottom',fontsize=8,color=INK,weight='bold')
    if ymax: ax.set_ylim(0,ymax)
    return b

# ==================================================================
pdf=PdfPages(R/'REPORTE_TESIS.pdf'); pg=0

# ---------- 1. PORTADA ----------
pg+=1; fig=plt.figure(figsize=A4);
fig.patch.set_facecolor('white')
ax=fig.add_axes([0,0,1,1]); ax.axis('off')
ax.add_patch(plt.Rectangle((0,.72),1,.28,color=TEAL,zorder=0))
fig.text(.5,.90,'REPORTE DE AVANCES',ha='center',fontsize=15,color='white',weight='bold',alpha=.9)
fig.text(.5,.83,'Sistema de conteo automatico de alevines',ha='center',fontsize=22,color='white',weight='bold')
fig.text(.5,.785,'Deep learning y vision artificial movil',ha='center',fontsize=13,color='white',alpha=.9)
fig.text(.5,.685,'CITE Productivo Madre de Dios · 2026',ha='center',fontsize=11,color=SOFT)
# KPIs
kpis=[('233','imagenes etiquetadas'),('0.836','mAP@50 (deteccion)'),
      ('97%','exactitud de conteo*'),('0.17%','error a nivel de lote')]
for i,(v,l) in enumerate(kpis):
    x=.09+i*.215
    ax.add_patch(FancyBboxPatch((x,.50),.19,.12,boxstyle='round,pad=.01',
        fc='#f2f7f7',ec=TEAL,lw=1.4,transform=fig.transFigure))
    fig.text(x+.095,.575,v,ha='center',fontsize=19,weight='bold',color=TEAL)
    fig.text(x+.095,.525,l,ha='center',fontsize=7.8,color=SOFT)
fig.text(.5,.475,'*mediana por imagen del mejor modelo',ha='center',fontsize=7.5,color=FAINT)
# resumen
txt=("Este reporte resume el desarrollo completo del sistema: captura de datos, tratamiento y\n"
     "etiquetado, entrenamiento y comparacion de modelos YOLO, evaluacion de deteccion y conteo,\n"
     "tracking para video, y el diseno de una estacion de captura estandarizada.")
fig.text(.5,.40,txt,ha='center',fontsize=10,color=SOFT,linespacing=1.6)
fig.text(.5,.09,'Generado automaticamente a partir de los resultados del repositorio',
         ha='center',fontsize=8,color=FAINT)
pdf.savefig(fig); plt.close(fig)

# ---------- 2. RESUMEN EJECUTIVO ----------
pg+=1; fig=plt.figure(figsize=A4); fig.patch.set_facecolor('white')
header(fig,'1','Resumen ejecutivo e hitos')
hitos=[
 ("Datos","257 fotos + 4 videos (2 sesiones: Android mayo, iPhone julio). 233 imagenes revisadas a mano, 20 701 alevines etiquetados."),
 ("Etiquetado colaborativo","Pre-etiquetado con IA (SAM/YOLO) + revision de 2 anotadores humanos, sincronizado por Git. Clase unica 'alevin'."),
 ("Metodologia rigurosa","Split POR VIDEO (evita fuga de datos), ground truth de conteo = nº de cajas, calibracion de umbral para conteo."),
 ("Comparacion de modelos","YOLOv8n/s y YOLOv11n/s/m a 960/1280/1536 px. Mejor deteccion: YOLOv11m@1536 (mAP@50=0.836)."),
 ("Contador general","Un solo modelo cuenta larvas y peces grandes (otra especie) en 2 sesiones distintas. Elimina el falso positivo del recipiente."),
 ("Tracking de video","BoT-SORT (sin compensacion de camara) supera a ByteTrack para contar por IDs unicos en camara fija."),
 ("Estacion de captura","Diseno de un entorno controlado (luz difusa + bandeja de una capa) que ataca cada fuente de error medida."),
]
y=.86
for t,d in hitos:
    fig.text(.09,y,'●',fontsize=12,color=TEAL)
    fig.text(.12,y,t,fontsize=11.5,weight='bold',color=INK)
    fig.text(.12,y-.028,d,fontsize=9.3,color=SOFT,wrap=True)
    y-=.088
# meta
fig.text(.09,.135,'Metrica clave para el CITE (conteo por LOTE):',fontsize=10,weight='bold',color=INK)
fig.text(.09,.105,'El sistema cuenta un lote completo con 0.17-2.6% de error — lo que importa al contar una tanda de alevines.',
         fontsize=9.5,color=SOFT)
footer(fig,pg); pdf.savefig(fig); plt.close(fig)

# ---------- 3. PIPELINE ----------
pg+=1; fig=plt.figure(figsize=A4); fig.patch.set_facecolor('white')
header(fig,'2','Pipeline del sistema (procesos)')
fases=[("1. Captura de datos","Videos y fotos cenitales con smartphone (4K), en distintas sesiones."),
 ("2. Extraccion y curacion","Extraccion de frames + deduplicacion perceptual (pHash) para quedarse con frames diversos."),
 ("3. Etiquetado","Pre-etiquetado automatico (IA) -> revision manual colaborativa en LabelImg -> ground truth de conteo."),
 ("4. Construccion del dataset","Split por video/temporal (sin leakage). Train / Val / Test + data augmentation."),
 ("5. Entrenamiento y ajuste","Transfer learning (COCO), comparacion de arquitecturas, resolucion e hiperparametros, calibracion de confianza."),
 ("6. Evaluacion","Deteccion (P, R, mAP) + conteo (MAE, RMSE, MAPE, R2, error de lote) + ablacion + generalizacion."),
 ("7. Resultado y despliegue","Modelo final -> conteo automatico. Tracking para video. Estacion de captura + export movil (.tflite).")]
y=.87
for i,(t,d) in enumerate(fases):
    ax=fig.add_axes([.09,y-.006,.05,.05]); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle='round,pad=.08',fc=TEAL,ec='none',
        transform=ax.transAxes)); ax.text(.5,.5,str(i+1),ha='center',va='center',color='white',weight='bold',fontsize=12)
    fig.text(.17,y+.022,t,fontsize=11,weight='bold',color=INK)
    fig.text(.17,y-.005,d,fontsize=9,color=SOFT)
    if i<len(fases)-1: fig.text(.113,y-.028,'↓',fontsize=11,color=TEAL,ha='center')
    y-=.107
footer(fig,pg); pdf.savefig(fig); plt.close(fig)

# ---------- 4. DATOS ----------
pg+=1; fig=plt.figure(figsize=A4); fig.patch.set_facecolor('white')
header(fig,'3','Tratamiento de datos')
# 4a evolucion dataset
ax1=fig.add_axes([.10,.60,.36,.26])
etapas=['Prelim.\n(mayo)','Completo\n(mayo)','General\n(+videos)']
imgs=[61,113,233]
bars(ax1,etapas,imgs,OKABE[0],fmt='{:.0f}')
ax1.set_title('Imagenes revisadas',fontsize=10.5,weight='bold',color=INK,pad=8)
ax1.set_ylim(0,270)
ax2=fig.add_axes([.56,.60,.36,.26])
boxes=[5363,17640,20701]
bars(ax2,etapas,boxes,OKABE[2],fmt='{:,.0f}')
ax2.set_title('Alevines etiquetados (cajas)',fontsize=10.5,weight='bold',color=INK,pad=8)
ax2.set_ylim(0,24000)
# 4b tamano de caja por fuente
ax3=fig.add_axes([.10,.20,.82,.26])
fuentes=['foto mayo','video mayo','IMG_0177\n(larvas)','IMG_0178\n(peces grandes)']
tam=[289,109,89,159]
b=bars(ax3,fuentes,tam,OKABE[5],fmt='{:.0f} px')
ax3.axhline(32,color=OKABE[3],lw=1.5,ls='--',zorder=4)
ax3.text(3.4,40,'32 px = "small" (COCO)',color=OKABE[3],fontsize=8,ha='right')
ax3.set_title('Tamano mediano del objeto por fuente (lado equivalente)',fontsize=10.5,weight='bold',color=INK,pad=8)
ax3.set_ylabel('pixeles'); ax3.set_ylim(0,320)
fig.text(.10,.13,'Hallazgo: los objetos NO son pequenos (todos > 32 px). El techo del mAP@50-95 no es el tamano,',fontsize=9,color=SOFT)
fig.text(.10,.108,'sino la generalizacion (ver pagina de la campana mAP@50-95).',fontsize=9,color=SOFT)
footer(fig,pg); pdf.savefig(fig); plt.close(fig)

# ---------- 5. COMPARACION DE MODELOS (233 test) ----------
pg+=1; fig=plt.figure(figsize=A4); fig.patch.set_facecolor('white')
header(fig,'4','Comparacion de modelos (test final, 233 imgs)')
g=load(R/'comparacion_general.csv'); gr=load(R/'comparacion_grandes.csv')
df=pd.concat([g,gr],ignore_index=True)
# nombres cortos
name_map={'YOLOv11n_general':'v11n@1280','YOLOv11s_general':'v11s@960','YOLOv8n_general':'v8n@960',
          'v11m_1280':'v11m@1280','v11n_1536':'v11n@1536','v11m_1536':'v11m@1536'}
df['nm']=df['modelo'].map(name_map).fillna(df['modelo'])
df=df.sort_values('mAP50_95')
x=np.arange(len(df)); w=.38
ax=fig.add_axes([.11,.55,.81,.33])
ax.bar(x-w/2,df['mAP50'],w,label='mAP@50',color=OKABE[0],zorder=3)
ax.bar(x+w/2,df['mAP50_95'],w,label='mAP@50-95',color=OKABE[1],zorder=3)
for xi,a,b in zip(x,df['mAP50'],df['mAP50_95']):
    ax.text(xi-w/2,a,f'{a:.2f}',ha='center',va='bottom',fontsize=7.5,color=INK)
    ax.text(xi+w/2,b,f'{b:.2f}',ha='center',va='bottom',fontsize=7.5,color=INK)
ax.set_xticks(x); ax.set_xticklabels(df['nm'],fontsize=8.5); ax.set_ylim(0,.95)
ax.legend(loc='upper left',frameon=False,fontsize=9)
ax.set_title('Metricas de deteccion por modelo',fontsize=10.5,weight='bold',color=INK,pad=8)
# tabla de conteo
ax2=fig.add_axes([.11,.12,.81,.32]); ax2.axis('off')
cols=['Modelo','mAP@50','mAP@50-95','MAPE','R2','Error lote']
rows=[]
for _,r in df.sort_values('mAP50_95',ascending=False).iterrows():
    rows.append([r['nm'],f"{r['mAP50']:.3f}",f"{r['mAP50_95']:.3f}",f"{r['MAPE']:.1f}%",f"{r['R2']:.3f}",f"{r['agg_pct']:.1f}%"])
t=ax2.table(cellText=rows,colLabels=cols,cellLoc='center',loc='center',bbox=[0,0,1,.95])
t.auto_set_font_size(False); t.set_fontsize(9)
for (ri,ci),cell in t.get_celld().items():
    cell.set_edgecolor('#d5dede')
    if ri==0: cell.set_facecolor(TEAL); cell.set_text_props(color='white',weight='bold')
    elif ri==1: cell.set_facecolor('#e3f1ee')
ax2.set_title('Resultados de conteo (mismo test)',fontsize=10.5,weight='bold',color=INK,y=.98)
fig.text(.11,.075,'Recomendado: YOLOv11n@1280 (equilibrio conteo/movil). Maxima exactitud: YOLOv11m@1536.',fontsize=9,color=SOFT)
footer(fig,pg); pdf.savefig(fig); plt.close(fig)

# ---------- 6. CONTEO (scatter + error) ----------
pg+=1; fig=plt.figure(figsize=A4); fig.patch.set_facecolor('white')
header(fig,'5','Resultado de conteo (modelo final)')
cf=load(C/'conteo_final_test.csv')
srcs=cf['source'].unique()
cmap={s:OKABE[i%len(OKABE)] for i,s in enumerate(sorted(srcs))}
ax=fig.add_axes([.11,.53,.5,.36])
lim=max(cf['real_count'].max(),cf['pred_count'].max())*1.08
ax.plot([0,lim],[0,lim],'--',color=FAINT,lw=1.3,label='ideal (y=x)',zorder=2)
for s in sorted(srcs):
    d=cf[cf['source']==s]
    ax.scatter(d['real_count'],d['pred_count'],s=55,color=cmap[s],edgecolor='white',lw=.8,label=s,zorder=3)
ax.set_xlabel('Conteo real (manual)'); ax.set_ylabel('Conteo automatico (YOLO)')
ax.set_xlim(0,lim); ax.set_ylim(0,lim); ax.legend(fontsize=7.5,frameon=False,loc='upper left')
ax.set_title('Real vs. predicho',fontsize=10.5,weight='bold',color=INK,pad=8)
re=cf['real_count'].values; pr=cf['pred_count'].values
r2=1-((pr-re)**2).sum()/((re-re.mean())**2).sum()
# panel metricas
ax2=fig.add_axes([.66,.53,.26,.36]); ax2.axis('off')
mets=[('R2',f'{r2:.3f}'),('MAPE (mediana)',f'{np.median(np.abs(pr-re)/np.maximum(re,1)*100):.1f}%'),
      ('Error de lote',f'{abs(pr.sum()-re.sum())/re.sum()*100:.1f}%'),
      ('Total real',f'{re.sum()}'),('Total predicho',f'{pr.sum()}')]
yy=.85
for k,v in mets:
    ax2.text(0,yy,k,fontsize=9,color=SOFT,transform=ax2.transAxes)
    ax2.text(1,yy,v,fontsize=11,color=TEAL,weight='bold',ha='right',transform=ax2.transAxes)
    yy-=.16
# MAPE por fuente
ax3=fig.add_axes([.11,.11,.81,.30])
by=cf.groupby('source').apply(lambda d:(np.abs(d['pred_count']-d['real_count'])/np.maximum(d['real_count'],1)*100).mean())
bars(ax3,list(by.index),list(by.values),OKABE[3],fmt='{:.1f}%',rot=15)
ax3.set_title('Error de conteo (MAPE) por fuente / condicion',fontsize=10.5,weight='bold',color=INK,pad=8)
ax3.set_ylabel('MAPE (%)')
footer(fig,pg); pdf.savefig(fig); plt.close(fig)

# ---------- 7. ABLACION + CAMPANA mAP@50-95 ----------
pg+=1; fig=plt.figure(figsize=A4); fig.patch.set_facecolor('white')
header(fig,'6','Ablacion de datos y techo del mAP@50-95')
# ablacion
ab=load(R/'ablacion_datos.csv')
ax=fig.add_axes([.11,.57,.36,.30])
ax.bar([0,1],[0.785,0.801],width=.5,color=[OKABE[6],OKABE[0]],zorder=3)
for xi,v in zip([0,1],[0.785,0.801]): ax.text(xi,v,f'{v:.3f}',ha='center',va='bottom',fontsize=9,weight='bold')
ax.set_xticks([0,1]); ax.set_xticklabels(['45 imgs','90 imgs']); ax.set_ylim(0,.9)
ax.set_title('Ablacion: mas datos -> mejor mAP@50',fontsize=10,weight='bold',color=INK,pad=8)
# diagnostico train vs test
ax2=fig.add_axes([.57,.57,.35,.30])
ax2.bar([0,1],[0.623,0.506],width=.5,color=[OKABE[2],OKABE[1]],zorder=3)
for xi,v in zip([0,1],[0.623,0.506]): ax2.text(xi,v,f'{v:.3f}',ha='center',va='bottom',fontsize=9,weight='bold')
ax2.axhline(0.60,color=OKABE[3],ls='--',lw=1.3); ax2.text(1.4,.61,'meta 0.60',color=OKABE[3],fontsize=8,ha='right')
ax2.set_xticks([0,1]); ax2.set_xticklabels(['TRAIN','TEST']); ax2.set_ylim(0,.75)
ax2.set_title('mAP@50-95: train vs test',fontsize=10,weight='bold',color=INK,pad=8)
# texto explicativo
fig.text(.11,.43,'Diagnostico clave:',fontsize=11,weight='bold',color=INK)
diag=("• Los modelos grandes + alta resolucion subieron el mAP@50-95 de 0.483 a 0.506 (mejora incremental).\n"
      "• En el TRAIN el modelo SI alcanza 0.62 -> las etiquetas soportan >0.60.\n"
      "• El limitante real es la BRECHA DE GENERALIZACION (train 0.62 -> test 0.51), no el modelo ni las etiquetas.\n"
      "• Conclusion: superar 0.60 en test es alcanzable con MAS DATOS DIVERSOS (mas sesiones/condiciones),\n"
      "  que es justamente el proximo paso del proyecto.")
fig.text(.11,.29,diag,fontsize=9.5,color=SOFT,linespacing=1.7)
footer(fig,pg); pdf.savefig(fig); plt.close(fig)

# ---------- 8. TRACKING ----------
pg+=1; fig=plt.figure(figsize=A4); fig.patch.set_facecolor('white')
header(fig,'7','Tracking de video: ByteTrack vs BoT-SORT')
tc=load(R/'trackers_comparacion.csv')
# fragmentacion por video y tracker
piv=tc.pivot_table(index='video',columns='tracker',values='frag_ratio')
trs=['bytetrack.yaml','botsort.yaml','trackers/botsort_static.yaml']
lbls=['ByteTrack','BoT-SORT','BoT-SORT estatico']
ax=fig.add_axes([.11,.55,.81,.33])
x=np.arange(len(piv.index)); w=.26
for i,(tr,lb) in enumerate(zip(trs,lbls)):
    vals=[piv.loc[v,tr] for v in piv.index]
    ax.bar(x+(i-1)*w,vals,w,label=lb,color=OKABE[i],zorder=3)
    for xi,vv in zip(x+(i-1)*w,vals): ax.text(xi,vv,f'{vv:.1f}',ha='center',va='bottom',fontsize=8)
ax.axhline(1,color=OKABE[2],ls='--',lw=1.3); ax.text(len(x)-.5,1.4,'ideal = 1.0',color=OKABE[2],fontsize=8,ha='right')
ax.set_xticks(x); ax.set_xticklabels(piv.index); ax.set_ylabel('Fragmentacion (menor = mejor)')
ax.legend(frameon=False,fontsize=9,loc='upper left')
ax.set_title('Fragmentacion de IDs (sobreconteo). Menor es mejor.',fontsize=10.5,weight='bold',color=INK,pad=8)
fig.text(.11,.42,'Conclusiones del tracking:',fontsize=11,weight='bold',color=INK)
tk=("• BoT-SORT supera claramente a ByteTrack (≈2x menos fragmentacion) — su Kalman mas preciso ayuda con objetos pequenos.\n"
    "• La compensacion de movimiento de camara (GMC) es inutil con tripode: misma calidad, 2x mas lento -> usar gmc=none.\n"
    "• A frame completo la fragmentacion baja de 4.3 a 1.89 (los peces casi no se mueven entre frames a 120 fps).\n"
    "• Recomendado: BoT-SORT con gmc_method=none, a frame completo.")
fig.text(.11,.29,tk,fontsize=9.5,color=SOFT,linespacing=1.7)
footer(fig,pg); pdf.savefig(fig); plt.close(fig)

# ---------- 9. CONCLUSIONES ----------
pg+=1; fig=plt.figure(figsize=A4); fig.patch.set_facecolor('white')
header(fig,'8','Conclusiones y proximos pasos')
fig.text(.09,.87,'Estado del sistema',fontsize=12,weight='bold',color=TEAL)
concl=("• Sistema funcional de extremo a extremo: captura -> etiquetado -> entrenamiento -> conteo.\n"
       "• Contador GENERAL validado en 2 sesiones/camaras: cuenta larvas y peces grandes, sin el falso positivo del recipiente.\n"
       "• A nivel de LOTE (lo que importa al CITE) el error es de 0.17-2.6%.\n"
       "• Metodologia solida y honesta: split sin fuga de datos, metricas del dominio, ablacion, analisis de fallos y de generalizacion.\n"
       "• Decidido el tracker de video (BoT-SORT) y disenada la estacion de captura estandarizada.")
fig.text(.09,.72,concl,fontsize=9.7,color=SOFT,linespacing=1.9)
fig.text(.09,.55,'Proximos pasos (por prioridad)',fontsize=12,weight='bold',color=TEAL)
nxt=("1. Mas datos diversos (2-3 sesiones nuevas con distinta luz/agua/recipiente) -> cerrar la brecha y superar 0.60 de mAP@50-95.\n"
     "2. Armar la estacion de captura estandarizada (luz difusa + bandeja de una capa) -> menos data, mas exactitud.\n"
     "3. Software de la estacion: capturar -> correr YOLO -> mostrar conteo, con bucle de mejora (active learning).\n"
     "4. Exportar a movil (.tflite/.onnx) y medir tamano/FPS en el dispositivo.\n"
     "5. (Opcional) Segmentacion/OBB para subir aun mas el techo del mAP@50-95.")
fig.text(.09,.38,nxt,fontsize=9.7,color=SOFT,linespacing=1.9)
fig.text(.09,.21,'Entregables generados',fontsize=12,weight='bold',color=TEAL)
ent=("• Repositorio con todo el pipeline (scripts reproducibles) y modelos entrenados.\n"
     "• Informe tecnico (docs/INFORME_TECNICO.md) con metodologia, resultados y referencias verificadas.\n"
     "• Diagramas: pipeline del sistema y especificacion de la estacion de captura.\n"
     "• Este reporte PDF + un notebook Jupyter ejecutable con todos los procesos.")
fig.text(.09,.07,ent,fontsize=9.7,color=SOFT,linespacing=1.9)
footer(fig,pg); pdf.savefig(fig); plt.close(fig)

pdf.close()
print(f'PDF generado: {R/"REPORTE_TESIS.pdf"} ({pg} paginas)')
