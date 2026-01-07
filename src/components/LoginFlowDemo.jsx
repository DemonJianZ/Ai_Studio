import React, { useMemo } from "react";
import {
  Sparkles,
  Image as ImageIcon,
  Video,
  Upload,
  FileText,
  Wand2,
  ArrowRight,
} from "lucide-react";

function Node({
  title,
  subtitle,
  icon: Icon,
  tone = "purple",
  x,
  y,
  // 调整：减小默认宽高，让整体视觉更精致，留白更多
  w = "clamp(140px, 18vw, 180px)",
  h = "clamp(68px, 9vh, 80px)",
}) {
  const toneMap = {
    purple: "from-purple-400/40 via-purple-400/10 to-transparent",
    cyan: "from-cyan-400/40 via-cyan-400/10 to-transparent",
    emerald: "from-emerald-400/40 via-emerald-400/10 to-transparent",
    amber: "from-amber-400/45 via-amber-400/10 to-transparent",
  };

  const desc =
    subtitle === "Input"
      ? "上传参考图或输入提示词，作为工作流起点。"
      : subtitle === "Text → Image"
      ? "根据 Prompt 生成商品主图 / 场景图。"
      : subtitle === "Image → Image"
      ? "保留主体，替换背景 / 风格迁移 / 细节修复。"
      : subtitle === "Image → Video"
      ? "以参考图为起点扩展为短视频片段。"
      : subtitle === "Output"
      ? "聚合结果，进入批量导出与发布。"
      : "工作流节点";

  return (
    <div
      className={[
        "absolute z-20 rounded-xl border border-white/10 bg-slate-900/35 backdrop-blur-xl", // rounded-2xl -> rounded-xl
        "shadow-[0_20px_60px_-40px_rgba(0,0,0,0.7)] overflow-hidden",
        "transition-transform duration-500 hover:scale-[1.02] hover:border-white/20",
      ].join(" ")}
      style={{
        left: `${x}%`,
        top: `${y}%`,
        width: w,
        height: h,
        transform: "translate(-50%, -50%)",
      }}
    >
      <div className={`absolute inset-0 bg-gradient-to-br ${toneMap[tone]} opacity-55 blur-2xl`} />
      <div className="absolute inset-0 bg-gradient-to-b from-white/6 via-transparent to-transparent" />

      {/* 调整：减小 padding (p-4 -> p-3) 和 gap */}
      <div className="relative h-full p-3 flex gap-2.5 items-center">
        {/* 调整：减小图标容器和图标尺寸 */}
        <div className="shrink-0 p-1.5 rounded-lg bg-slate-950/40 border border-white/10">
          <Icon className="w-3.5 h-3.5 text-white" />
        </div>
        <div className="min-w-0 flex flex-col justify-center">
          {/* 调整：减小字号 */}
          <div className="text-[9px] text-slate-400 font-semibold uppercase tracking-[0.18em] truncate">
            {subtitle}
          </div>
          <div className="text-xs font-bold text-white truncate leading-tight my-0.5">{title}</div>
          <div className="text-[9px] text-slate-400 leading-relaxed line-clamp-2 transform scale-95 origin-top-left">
            {desc}
          </div>
        </div>
      </div>
    </div>
  );
}

function Dot({ pathId, delay = "0s" }) {
  return (
    <circle r="3.1" fill="white" opacity="0.95">
      <animateMotion dur="5.2s" repeatCount="indefinite" begin={delay}>
        <mpath href={`#${pathId}`} />
      </animateMotion>
    </circle>
  );
}

export default function LoginFlowDemo() {
  const layout = useMemo(() => {
    /**
     * 调整后的布局策略：
     * 1. X轴采用更均匀的分布 (16% -> 50% -> 84%)，留出更多呼吸感
     * 2. Y轴保持垂直居中对称，Processors 在 20/50/80 分布，Inputs 穿插在 35/65
     */
    const nodes = {
      // Column 1: Inputs
      prompt:    { x: 16, y: 35 },
      inputImg:  { x: 16, y: 65 },

      // Column 2: Processors (Center Column)
      t2i: { x: 50, y: 20 },
      i2i: { x: 50, y: 50 },
      i2v: { x: 50, y: 80 },

      // Column 3: Output
      output: { x: 84, y: 50 },
    };

    const map = (p) => ({
      x: (p.x / 100) * 1000,
      y: (p.y / 100) * 560,
    });

    const A = Object.fromEntries(Object.entries(nodes).map(([k, v]) => [k, map(v)]));

    // 锚点偏移量
    const leftOut = 125;  // 节点右侧出线
    const midOut = 125;   // 中间节点出线
    const inToMid = 125;  // 进入中间节点
    const inToOut = 125;  // 进入输出节点

    // 贝塞尔曲线控制点偏移 (控制线的弯曲程度)
    const c1 = 180; 

    const paths = [
      {
        id: "p_prompt_t2i",
        d: `M ${A.prompt.x + leftOut} ${A.prompt.y}
            C ${A.prompt.x + leftOut + c1} ${A.prompt.y},
              ${A.t2i.x - inToMid - c1} ${A.t2i.y},
              ${A.t2i.x - inToMid} ${A.t2i.y}`,
      },
      {
        id: "p_in_i2i",
        d: `M ${A.inputImg.x + leftOut} ${A.inputImg.y}
            C ${A.inputImg.x + leftOut + c1} ${A.inputImg.y},
              ${A.i2i.x - inToMid - c1} ${A.i2i.y},
              ${A.i2i.x - inToMid} ${A.i2i.y}`,
      },
      {
        id: "p_in_i2v",
        d: `M ${A.inputImg.x + leftOut} ${A.inputImg.y}
            C ${A.inputImg.x + leftOut + c1} ${A.inputImg.y},
              ${A.i2v.x - inToMid - c1} ${A.i2v.y},
              ${A.i2v.x - inToMid} ${A.i2v.y}`,
      },
      {
        id: "p_t2i_out",
        d: `M ${A.t2i.x + midOut} ${A.t2i.y}
            C ${A.t2i.x + midOut + c1} ${A.t2i.y},
              ${A.output.x - inToOut - c1} ${A.output.y},
              ${A.output.x - inToOut} ${A.output.y}`,
      },
      {
        id: "p_i2i_out",
        d: `M ${A.i2i.x + midOut} ${A.i2i.y}
            C ${A.i2i.x + midOut + c1} ${A.i2i.y},
              ${A.output.x - inToOut - c1} ${A.output.y},
              ${A.output.x - inToOut} ${A.output.y}`,
      },
      {
        id: "p_i2v_out",
        d: `M ${A.i2v.x + midOut} ${A.i2v.y}
            C ${A.i2v.x + midOut + c1} ${A.i2v.y},
              ${A.output.x - inToOut - c1} ${A.output.y},
              ${A.output.x - inToOut} ${A.output.y}`,
      },
    ];

    return { nodes, paths };
  }, []);

  return (
    <div className="relative h-full w-full rounded-3xl border border-white/10 overflow-hidden bg-gradient-to-br from-slate-950 via-slate-950/80 to-slate-900/60 shadow-2xl shadow-purple-900/25">
      {/* ambient gradients */}
      <div
        className="absolute inset-0 opacity-60"
        style={{
          backgroundImage:
            "radial-gradient(circle at 18% 22%, rgba(124,58,237,0.22), transparent 42%)," +
            "radial-gradient(circle at 82% 18%, rgba(34,211,238,0.18), transparent 40%)," +
            "radial-gradient(circle at 60% 86%, rgba(16,185,129,0.14), transparent 44%)",
        }}
      />

      {/* grid overlay */}
      <div
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage:
            "linear-gradient(transparent 95%, rgba(255,255,255,0.10) 96%)," +
            "linear-gradient(90deg, transparent 95%, rgba(255,255,255,0.10) 96%)",
          backgroundSize: "28px 28px, 28px 28px",
        }}
      />

      <div className="absolute top-6 right-8 z-30 flex items-center gap-2 text-[10px] font-bold tracking-[0.22em] text-slate-400">
        <Sparkles className="w-3.5 h-3.5 text-purple-400" />
        <span className="uppercase">Workflow Preview</span>
      </div>

      {/* inner canvas */}
      <div className="absolute inset-0 rounded-2xl border border-white/5 bg-slate-900/18 backdrop-blur-sm shadow-inner overflow-hidden">
        {/* connectors under nodes */}
        <svg viewBox="0 0 1000 560" className="absolute inset-0 w-full h-full z-10">
          <defs>
            <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.42" />
              <stop offset="55%" stopColor="#a855f7" stopOpacity="0.52" />
              <stop offset="100%" stopColor="#fbbf24" stopOpacity="0.40" />
            </linearGradient>

            <linearGradient id="flowGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgba(255,255,255,0)" />
              <stop offset="45%" stopColor="rgba(255,255,255,0.65)" />
              <stop offset="55%" stopColor="rgba(255,255,255,0.28)" />
              <stop offset="100%" stopColor="rgba(255,255,255,0)" />
            </linearGradient>
          </defs>

          {layout.paths.map((p, idx) => (
            <path
              key={`base_${p.id}`}
              id={p.id}
              d={p.d}
              fill="none"
              stroke="url(#lineGrad)"
              strokeWidth="2"
              opacity="0.35"
              className="bf-line"
              style={{ animationDelay: `${idx * 0.25}s` }}
            />
          ))}

          {layout.paths.map((p, idx) => (
            <path
              key={`flow_${p.id}`}
              d={p.d}
              fill="none"
              stroke="url(#flowGrad)"
              strokeWidth="4"
              opacity="0.55"
              className="bf-flow"
              style={{ animationDelay: `${idx * 0.35}s` }}
            />
          ))}

          <g style={{ filter: "drop-shadow(0 0 8px rgba(168,85,247,0.45))" }}>
            <Dot pathId="p_prompt_t2i" delay="0s" />
            <Dot pathId="p_in_i2i" delay="0.6s" />
            <Dot pathId="p_in_i2v" delay="1.2s" />
            <Dot pathId="p_t2i_out" delay="0.4s" />
            <Dot pathId="p_i2i_out" delay="1.0s" />
            <Dot pathId="p_i2v_out" delay="1.6s" />
          </g>
        </svg>

        {/* nodes */}
        <Node title="Prompt" subtitle="Input" icon={FileText} tone="cyan" x={layout.nodes.prompt.x} y={layout.nodes.prompt.y} />
        <Node title="上传图片" subtitle="Input" icon={Upload} tone="cyan" x={layout.nodes.inputImg.x} y={layout.nodes.inputImg.y} />

        <Node title="文生图" subtitle="Text → Image" icon={Wand2} tone="purple" x={layout.nodes.t2i.x} y={layout.nodes.t2i.y} />
        <Node title="图生图" subtitle="Image → Image" icon={ImageIcon} tone="emerald" x={layout.nodes.i2i.x} y={layout.nodes.i2i.y} />
        <Node title="图生视频" subtitle="Image → Video" icon={Video} tone="amber" x={layout.nodes.i2v.x} y={layout.nodes.i2v.y} />

        <Node title="输出" subtitle="Output" icon={ArrowRight} tone="purple" x={layout.nodes.output.x} y={layout.nodes.output.y} />

        {/* noise */}
        <div className="absolute inset-0 pointer-events-none opacity-15 bf-noise z-0" />
      </div>

      <style>{`
        .bf-line { animation: bfLine 3.6s ease-in-out infinite; }
        @keyframes bfLine { 0%,100% { opacity: 0.28; } 50% { opacity: 0.48; } }

        .bf-flow {
          stroke-dasharray: 18 240;
          stroke-dashoffset: 0;
          animation: bfFlow 2.8s linear infinite;
        }
        @keyframes bfFlow {
          0% { stroke-dashoffset: 0; opacity: 0.25; }
          35% { opacity: 0.85; }
          100% { stroke-dashoffset: -260; opacity: 0.25; }
        }

        .bf-noise {
          background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
        }
      `}</style>
    </div>
  );
}