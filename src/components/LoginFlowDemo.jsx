import { motion } from "framer-motion";
import { Cpu, Database, Download, Sparkles, Workflow } from "lucide-react";

const nodePulseTransition = (delay = 0) => ({
  duration: 1.1,
  delay,
  repeat: Infinity,
  repeatDelay: 5.2,
  ease: [0.22, 1, 0.36, 1],
});

const floatTransition = {
  duration: 6.5,
  repeat: Infinity,
  repeatType: "mirror",
  ease: "easeInOut",
};

const nodes = [
  {
    id: "input",
    title: "Dataset Intake",
    subtitle: "Images & Prompts",
    icon: Database,
    delay: 0.15,
    position: "left-6 top-10",
    accent: "from-cyan-400/50 via-cyan-400/20 to-transparent",
    body: (
      <div className="text-[11px] text-slate-300/90">拖拽素材、文案与参考图像</div>
    ),
  },
  {
    id: "process",
    title: "Processor",
    subtitle: "Nodes x4",
    icon: Cpu,
    delay: 0.6,
    position: "left-1/2 -translate-x-1/2 top-32",
    accent: "from-purple-400/60 via-purple-400/20 to-transparent",
    body: (
      <div className="flex items-center gap-2 text-[11px] text-purple-100/90">
        <div className="relative w-11 h-11">
          <svg viewBox="0 0 44 44" className="w-full h-full">
            <circle
              cx="22"
              cy="22"
              r="18"
              className="stroke-slate-700"
              strokeWidth="4"
              fill="none"
            />
            <motion.circle
              cx="22"
              cy="22"
              r="18"
              strokeWidth="4"
              stroke="url(#processorGradient)"
              fill="none"
              strokeLinecap="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: [0.15, 0.85, 0.15] }}
              transition={{ duration: 3.6, repeat: Infinity, ease: "easeInOut" }}
            />
            <defs>
              <linearGradient id="processorGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#c084fc" />
                <stop offset="50%" stopColor="#38bdf8" />
                <stop offset="100%" stopColor="#a855f7" />
              </linearGradient>
            </defs>
          </svg>
          <motion.div
            className="absolute inset-1 rounded-full bg-slate-900/80 border border-purple-500/20 flex items-center justify-center text-[10px] text-purple-100"
            animate={{ rotate: [0, 12, -8, 0] }}
            transition={{ duration: 3.6, repeat: Infinity, ease: "easeInOut" }}
          >
            循环处理
          </motion.div>
        </div>
        <div className="leading-tight">
          <div className="font-semibold text-sm">迭代加速中</div>
          <div className="text-[10px] text-slate-300">多节点并发 · QoS 平衡</div>
        </div>
      </div>
    ),
  },
  {
    id: "logic",
    title: "Logic",
    subtitle: "Routing",
    icon: Workflow,
    delay: 1.05,
    position: "right-6 top-14",
    accent: "from-emerald-400/60 via-emerald-400/25 to-transparent",
    body: (
      <div className="text-[11px] text-emerald-50/90">规则切换、路径分支与监控</div>
    ),
  },
  {
    id: "output",
    title: "Output",
    subtitle: "Delivery",
    icon: Download,
    delay: 1.5,
    position: "right-10 bottom-8",
    accent: "from-amber-400/70 via-amber-400/25 to-transparent",
    body: (
      <div className="flex items-center gap-2 text-[11px] text-amber-50/95">
        <div className="w-2.5 h-2.5 rounded-full bg-amber-300 animate-ping" />
        <div className="animate-pulse">下行中 · CDN 分发</div>
      </div>
    ),
  },
];

const connectorConfig = [
  { id: "path-1", d: "M132 86 C 170 90 188 94 214 82" },
  { id: "path-2", d: "M214 126 C 228 148 240 178 232 214" },
  { id: "path-3", d: "M140 182 C 156 190 172 200 212 210" },
];

function NodeCard({ title, subtitle, icon: Icon, accent, body, delay, position }) {
  return (
    <motion.div
      className={`group absolute ${position} w-60 rounded-2xl border border-slate-800/60 bg-slate-900/70 backdrop-blur-xl p-4 shadow-[0_20px_70px_-35px_rgba(0,0,0,0.6)] overflow-hidden`}
      initial={{ opacity: 0, scale: 0.92, y: 16 }}
      animate={{
        opacity: [0, 1, 1],
        scale: [0.92, 1, 1],
        y: [16, 0, 0],
      }}
      transition={nodePulseTransition(delay)}
    >
      <motion.div
        className="absolute inset-px rounded-2xl"
        animate={{ opacity: [0.3, 0.6, 0.3] }}
        transition={{ duration: 6.6, repeat: Infinity, ease: "easeInOut" }}
        style={{
          backgroundImage: `linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.06) 50%, transparent 100%)`,
        }}
      />
      <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${accent} opacity-40 blur-2xl group-hover:opacity-60 transition`} />
      <div className="relative flex items-center gap-3">
        <div className="p-2 rounded-xl bg-slate-800/70 border border-slate-700/60 group-hover:border-white/40 group-hover:shadow-[0_0_25px_-12px_rgba(255,255,255,0.6)] transition">
          <Icon className="w-4 h-4 text-white" />
        </div>
        <div className="leading-tight">
          <div className="text-sm text-slate-300">{subtitle}</div>
          <div className="font-semibold text-white">{title}</div>
        </div>
      </div>
      <div className="relative mt-3">{body}</div>
    </motion.div>
  );
}

export default function LoginFlowDemo() {
  return (
    <motion.div
      className="relative h-full min-h-[420px] rounded-2xl border border-slate-800/60 bg-gradient-to-br from-slate-950 via-slate-950/80 to-slate-900/70 overflow-hidden shadow-2xl shadow-purple-900/40"
      animate={{ y: [0, -3, 0] }}
      transition={floatTransition}
    >
      <div
        className="absolute inset-0 opacity-60"
        style={{
          backgroundImage:
            "radial-gradient(circle at 20% 30%, rgba(124,58,237,0.15), transparent 28%),radial-gradient(circle at 80% 20%, rgba(34,211,238,0.18), transparent 24%),radial-gradient(circle at 60% 80%, rgba(16,185,129,0.12), transparent 26%)",
        }}
      />
      <div
        className="absolute inset-0 mix-blend-soft-light"
        style={{
          backgroundImage:
            "linear-gradient(transparent 95%, rgba(148,163,184,0.18) 96%), linear-gradient(90deg, transparent 95%, rgba(148,163,184,0.18) 96%)",
          backgroundSize: "24px 24px, 24px 24px",
        }}
      />

      <div className="absolute top-4 right-4 z-20 flex items-center gap-2 text-xs text-slate-200/80">
        <Sparkles className="w-4 h-4 text-purple-200" />
        <span className="tracking-wide uppercase">AI Workflow in Motion</span>
      </div>

      <motion.div
        className="relative h-full"
        animate={{ y: [2, -2, 2] }}
        transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }}
      >
        <div className="absolute inset-0 px-4 pt-8 pb-6">
          <div className="relative h-full w-full rounded-2xl border border-slate-800/70 bg-slate-900/40 backdrop-blur-xl shadow-inner overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-b from-white/3 via-transparent to-transparent pointer-events-none" />

            <motion.svg
              viewBox="0 0 320 240"
              className="absolute inset-0 w-full h-full"
              initial={{ opacity: 0 }}
              animate={{ opacity: [0, 1, 1] }}
              transition={nodePulseTransition(0.3)}
            >
              {connectorConfig.map((conn, idx) => (
                <g key={conn.id}>
                  <motion.path
                    id={conn.id}
                    d={conn.d}
                    fill="none"
                    stroke="url(#lineGradient)"
                    strokeWidth={2.6}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeDasharray="6 8"
                    initial={{ pathLength: 0, strokeDashoffset: 1 }}
                    animate={{ pathLength: [0, 1, 1], strokeDashoffset: [1, 0, 0] }}
                    transition={{ duration: 1.4, delay: 0.6 + idx * 0.45, repeat: Infinity, repeatDelay: 4.5, ease: "easeInOut" }}
                    className="drop-shadow-[0_0_12px_rgba(168,85,247,0.35)]"
                  />
                  <circle r={4} fill="url(#lineGradient)" className="shadow-[0_0_12px_rgba(96,165,250,0.4)]">
                    <animateMotion dur="7s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear" begin={`${0.6 + idx * 0.45}s`}>
                      <mpath href={`#${conn.id}`} />
                    </animateMotion>
                  </circle>
                </g>
              ))}

              <defs>
                <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#38bdf8" />
                  <stop offset="50%" stopColor="#a855f7" />
                  <stop offset="100%" stopColor="#fbbf24" />
                </linearGradient>
              </defs>
            </motion.svg>

            {nodes.map((node) => (
              <NodeCard key={node.id} {...node} />
            ))}
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
