#!/usr/bin/env python3
"""Rebuild two vector figures from the accompanying verified data extract.

Run: python make_figures.py
Requires numpy and matplotlib. No model inference, training, or new statistics.
"""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "figure_data.json").read_text())
INK, MUTED, LINE = "#172B42", "#55677B", "#D7E0E8"
TEAL, BLUE, ORANGE = "#007F83", "#2A78D6", "#D45E2E"
PALE, PALETEAL = "#F5F8FB", "#EDF7F6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none",
    "text.color": INK, "axes.labelcolor": INK,
    "axes.unicode_minus": True, "savefig.facecolor": "white",
})

def canvas(w, h):
    fig = plt.figure(figsize=(w, h), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, w), ylim=(0, h)); ax.axis("off")
    return fig, ax

def text(ax, x, y, s, size=11, weight="normal", color=INK, ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                   ha=ha, va=va, **kw)

def box(ax, x, y, w, h, face=PALE, edge=LINE, radius=.07, lw=.9):
    p = FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=face,edgecolor=edge,linewidth=lw)
    ax.add_patch(p); return p

def arrow(ax, a, b, color=MUTED, lw=1.2, style="-|>", connection="arc3,rad=0"):
    p=FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=10,
                      linewidth=lw,color=color,connectionstyle=connection,
                      shrinkA=0,shrinkB=0)
    ax.add_patch(p); return p

def route(ax, points, color=MUTED, lw=1.1, dash=None):
    xs,ys=zip(*points)
    ax.plot(xs[:-1],ys[:-1],color=color,lw=lw,ls=dash or "-",clip_on=False)
    arrow(ax,points[-2],points[-1],color=color,lw=lw)

def export(fig, stem):
    meta={"Title": stem.replace("_", " "), "Author": "",
          "Subject": "EUOT-VPR scientific figure; verified source data",
          "Keywords": "visual place recognition, optimal transport, normalization"}
    fig.savefig(HERE / f"{stem}.pdf", metadata=meta)
    fig.savefig(HERE / f"{stem}.svg")
    fig.savefig(HERE / f"{stem}.png", dpi=600)
    plt.close(fig)

def make_teaser():
    fig,ax=canvas(9.0,3.98)
    text(ax,.25,3.73,"(a) Column-scale cancellation",11.5,"bold")
    text(ax,4.75,3.73,"(b) Combined-gate results",11.5,"bold")
    ax.plot([4.48,4.48],[.57,3.51],color=LINE,lw=.9)

    # Exact algebra: q_j is defined explicitly below, not measured data.
    box(ax,.25,2.37,1.68,1.03)
    text(ax,1.09,3.18,"POOLED CLUSTER",10.3,"bold",ha="center",color=MUTED)
    text(ax,1.09,2.78,r"$\tilde{\mathbf{v}}_j=b_j\,\mathbf{q}_j$",13,ha="center")
    arrow(ax,(1.98,2.86),(2.40,2.86))
    text(ax,2.19,3.09,r"$\ell_2$",9.7,ha="center",color=MUTED)
    box(ax,2.45,2.37,1.78,1.03,face=PALETEAL,edge="#B9DDDA")
    text(ax,3.34,3.18,"UNIT CLUSTER",10.3,"bold",ha="center",color=TEAL)
    text(ax,3.34,2.78,r"$\mathbf{v}_j=\frac{\mathbf{q}_j}{\|\mathbf{q}_j\|_2}$",14,ha="center")
    text(ax,.32,2.10,r"Direct column scaling $b_j$ cancels.",11.4,"bold",color=TEAL)
    text(ax,.32,1.87,"Indirect effects through row scaling remain.",10.5,color=MUTED)
    box(ax,.25,.90,3.98,.70,face=PALETEAL,edge="#B9DDDA")
    text(ax,.42,1.39,"MASS-ONLY GATE (ILLUSTRATION)",10.3,"bold",color=TEAL)
    text(ax,.43,1.12,r"$m_j=\sum_iP_{ij}$",12)
    arrow(ax,(1.92,1.12),(2.22,1.12),color=TEAL)
    text(ax,2.37,1.12,r"$\mathbf{v}_j^{\mathrm{agg}}=m_j\,\mathbf{v}_j$",12)
    text(ax,.29,.65,r"$\mathbf{q}_j=\sum_i a_iK_{ij}(\mathbf{f}_i-\alpha_p\mathbf{w}_j)$",10.8)
    text(ax,.29,.23,r"Assumes $b_j>0$ and $\tilde{\mathbf{v}}_j\neq\mathbf{0}$.",10.2,color=MUTED)

    text(ax,4.77,3.40,"MSLS-val  |  50% target masking",10.1,color=MUTED)
    plot=fig.add_axes([5.40/9,.90/3.98,2.38/9,2.18/3.98])
    plot.set_xlim(-6.6,6.0); plot.set_ylim(-.55,3.55)
    plot.axhspan(1.5,3.55,color="#F0F5FD",zorder=0)
    plot.axhspan(-.55,1.5,color="#FCF3EF",zorder=0)
    plot.axvline(0,color=MUTED,lw=1.0,zorder=1)
    settings=[("U",3,"o",BLUE,True),("RE",2,"s",BLUE,False),
              ("UW",1,"o",ORANGE,True),("S",0,"s",ORANGE,False)]
    for key,y,mk,col,filled in settings:
        d=DATA["interaction_at_50_percent_masking"][key]
        v=d["diff_pp"]; lo,hi=d["ci95"]
        plot.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt=mk,color=col,
                      mfc=col if filled else "white",mec=col,mew=1.3,
                      markersize=6.7,elinewidth=1.7,capsize=3.0,zorder=3)
        plot.text(7.1,y,f"{v:+.2f}",va="center",ha="left",
                  fontsize=11.3,fontweight="bold",color=col,clip_on=False)
    plot.set_xticks([-6,-3,0,3,6])
    plot.set_yticks([3,2,1,0])
    plot.set_yticklabels([r"$\mathrm{U}_{1/2}$",r"$\mathrm{R}_{1/2}$",
                         r"$\mathrm{U}_{1/65}$",r"$\mathrm{R}_{1/65}$"],fontsize=11.4)
    plot.tick_params(axis="both",length=0,labelcolor=MUTED,pad=6)
    plot.tick_params(axis="x",labelsize=10.4)
    for sp in plot.spines.values():sp.set_visible(False)
    plot.grid(axis="x",color=LINE,lw=.55,zorder=0)
    text(ax,8.02,3.26,"pp",10.0,color=MUTED)
    text(ax,4.76,.53,"Transport × gate interaction (pp)",10.8)
    text(ax,4.76,.23,"U: unit · R: raw  |  Whiskers: 95% CIs",10.2,color=MUTED)
    export(fig,"fig_teaser")

def make_method():
    fig,ax=canvas(9.0,6.0)
    text(ax,.23,5.73,"A matched test of the mass channel",17.5,"bold")
    text(ax,.25,5.42,"Real-image study: frozen SALAD trunk and projections; independently trained heads",
         10.4,color=MUTED)
    # Bypass routes preserve the actual data dependencies of the implementation.
    route(ax,[(1.03,4.80),(1.03,5.10),(5.55,5.10),(5.55,4.80)],color=MUTED)
    text(ax,3.08,5.15,"unit or raw token features",10.0,ha="center",color=MUTED,
         bbox={"facecolor":"white","edgecolor":"none","pad":1.4})
    xs=[.25,2.21,4.61,6.87]; ws=[1.56,2.03,1.87,1.88]
    for i,(x,w) in enumerate(zip(xs,ws)):
        box(ax,x,3.29,w,1.51,face=PALE if i<3 else PALETEAL,
            edge=LINE if i<3 else "#B9DDDA")
    titles=["Shared features","Reliability + transport","Pool + normalize","Gate + descriptor"]
    for x,w,t in zip(xs,ws,titles):text(ax,x+w/2,4.57,t,10.5,"bold",ha="center")
    text(ax,1.03,4.23,r"$322\times322$ image",10.8,ha="center")
    text(ax,1.03,3.93,r"$N=529,\ D=128$",10.8,ha="center")
    text(ax,1.03,3.63,"64 clusters + dustbin",9.9,ha="center")
    text(ax,1.03,3.42,r"Outputs: $\mathbf{f},\ \mathbf{S},\ \mathbf{g}$",10.5,ha="center")
    text(ax,3.225,4.22,r"$r_i$: learned reliability",10.5,ha="center")
    text(ax,3.225,3.94,r"$\mu_i=(0.15+0.85r_i)/N$",10.7,ha="center")
    text(ax,3.225,3.66,r"Dustbin boost: $\beta(1-r_i)$",10.0,ha="center")
    text(ax,3.225,3.42,"Six log-domain scaling cycles",9.9,ha="center")
    text(ax,5.545,4.22,r"Selected tokens: $\mathbf{x}_i$",10.4,ha="center")
    text(ax,5.545,3.92,r"$\tilde{\mathbf{v}}_j=\sum_iP_{ij}\mathbf{x}_i$",11.5,ha="center")
    text(ax,5.545,3.55,r"$\mathbf{v}_j=\tilde{\mathbf{v}}_j/\|\tilde{\mathbf{v}}_j\|_2$",10.9,ha="center")
    text(ax,7.81,4.22,r"Off: $\mathbf{v}_j$   On: $\psi_j\tilde{\mathbf{v}}_j$",10.8,ha="center")
    text(ax,7.81,3.94,r"Normalize cluster block $\mathbf{c}$",9.8,ha="center")
    text(ax,7.81,3.60,r"$\mathbf{D}=\mathrm{norm}([\mathbf{c};\alpha\mathbf{g}])$",10.8,ha="center")
    for a,b in [(1.81,2.21),(4.24,4.61),(6.48,6.87)]:
        arrow(ax,(a+.04,4.03),(b-.04,4.03))
    text(ax,4.425,4.24,r"$P$",10.6,ha="center",color=MUTED)
    route(ax,[(1.03,3.29),(1.03,2.98),(7.81,2.98),(7.81,3.29)],color=MUTED)
    text(ax,4.50,2.98,r"unit global token $\mathbf{g}$",10.1,ha="center",
         color=MUTED,bbox={"facecolor":"white","edgecolor":"none","pad":1.5})

    text(ax,.25,2.60,"(a) Column condition × gate",11.6,"bold")
    text(ax,4.92,2.60,"(b) Token choice × global share",11.6,"bold")
    # Arms: no claim of full row-and-column balance.
    x=.25;y=.72;w=4.18;h=1.58
    box(ax,x,y,w,h,face="white")
    rowlabel=1.74; cw=(w-rowlabel)/2
    ax.add_patch(Rectangle((x+rowlabel,y),cw,h,facecolor="#F7F9FB",edgecolor="none"))
    ax.add_patch(Rectangle((x+rowlabel+cw,y),cw,h,facecolor=PALETEAL,edgecolor="none"))
    ax.plot([x,x+w],[y+.57,y+.57],color=LINE,lw=.8)
    ax.plot([x,x+w],[y+1.14,y+1.14],color=LINE,lw=.8)
    for xx in [x+rowlabel,x+rowlabel+cw]:ax.plot([xx,xx],[y,y+h],color=LINE,lw=.8)
    text(ax,x+rowlabel+cw/2,y+1.37,"Gate off",10.8,"bold",ha="center")
    text(ax,x+rowlabel+1.5*cw,y+1.37,"Combined gate",10.3,"bold",ha="center",color=TEAL)
    text(ax,x+.12,y+.86,r"Balanced: $\rho_2=1$",10.6)
    text(ax,x+.12,y+.29,r"Relaxed: learned $\rho_2$",10.1)
    for xx,yy,s,c in [(x+rowlabel+cw/2,y+.86,"Bal",INK),
                       (x+rowlabel+1.5*cw,y+.86,"Bal+G",TEAL),
                       (x+rowlabel+cw/2,y+.29,"UOT",INK),
                       (x+rowlabel+1.5*cw,y+.29,"UOT+G",TEAL)]:
        text(ax,xx,yy,s,12.4,"bold",ha="center",color=c)
    text(ax,.29,.47,"Both column conditions retain row relaxation.",10.1,color=MUTED)
    text(ax,.29,.25,r"Reliability factor: $\psi_j=\bar r_j^2/(\bar r_j^2+0.18^2)$.",10.1,color=MUTED)
    # Exact descriptor-setting mapping.
    x=4.92;y=.72;w=3.83;h=1.58;rowlabel=1.06;cw=(w-rowlabel)/2
    box(ax,x,y,w,h,face="white")
    ax.add_patch(Rectangle((x+rowlabel,y),cw,h,facecolor="#F0F5FD",edgecolor="none"))
    ax.add_patch(Rectangle((x+rowlabel+cw,y),cw,h,facecolor="#FCF3EF",edgecolor="none"))
    ax.plot([x,x+w],[y+.57,y+.57],color=LINE,lw=.8)
    ax.plot([x,x+w],[y+1.14,y+1.14],color=LINE,lw=.8)
    for xx in [x+rowlabel,x+rowlabel+cw]:ax.plot([xx,xx],[y,y+h],color=LINE,lw=.8)
    text(ax,x+rowlabel+cw/2,y+1.37,r"Share $1/2$",10.7,"bold",ha="center",color=BLUE)
    text(ax,x+rowlabel+1.5*cw,y+1.37,r"Share $1/65$",10.7,"bold",ha="center",color=ORANGE)
    text(ax,x+.12,y+.86,r"Unit $\hat{\mathbf{f}}_i$",10.5)
    text(ax,x+.12,y+.29,r"Raw $\mathbf{f}_i$",10.5)
    for xx,yy,s,c in [(x+rowlabel+cw/2,y+.86,r"$\mathrm{U}_{1/2}$",BLUE),
                       (x+rowlabel+1.5*cw,y+.86,r"$\mathrm{U}_{1/65}$",ORANGE),
                       (x+rowlabel+cw/2,y+.29,r"$\mathrm{R}_{1/2}$",BLUE),
                       (x+rowlabel+1.5*cw,y+.29,r"$\mathrm{R}_{1/65}$",ORANGE)]:
        text(ax,xx,yy,s,13.2,"bold",ha="center",color=c)
    text(ax,4.96,.47,r"Global share $=\alpha^2/(1+\alpha^2)$.",10.2,color=MUTED)
    text(ax,4.96,.25,r"$\alpha=1$ for $1/2$; $\alpha=1/8$ for $1/65$.",10.2,color=MUTED)
    export(fig,"fig_matched_head_design")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("teaser", "method", "both"), default="both")
    selected = parser.parse_args().figure
    if selected in ("teaser", "both"):
        make_teaser()
    if selected in ("method", "both"):
        make_method()
