import React, { useMemo, useState } from "react";
import { useNavigate, Link } from "../router";
import { Zap, Shield, Mail, Lock, Loader2, CheckCircle2 } from "lucide-react";
import { useAuth } from "../auth/AuthProvider";
import LoginFlowDemo from "../components/LoginFlowDemo";

const gradientBg =
  "bg-[radial-gradient(circle_at_20%_20%,rgba(147,51,234,0.12),transparent_40%),radial-gradient(circle_at_80%_80%,rgba(56,189,248,0.08),transparent_40%)]";

const inputStyle =
  "w-full bg-slate-900/40 border border-slate-800/60 rounded-xl px-4 py-3 text-sm text-slate-100 " +
  "placeholder:text-slate-600 focus:outline-none focus:border-purple-500/50 focus:ring-4 focus:ring-purple-500/10 " +
  "transition-all duration-300";

const ENTERPRISE_DOMAIN = "dayukeji.com";
const PUBLIC_EMAIL_DOMAINS = new Set([
  "gmail.com",
  "outlook.com",
  "hotmail.com",
  "live.com",
  "yahoo.com",
  "icloud.com",
  "qq.com",
  "163.com",
  "126.com",
  "proton.me",
  "yeah.net",
]);

function getEmailDomain(email) {
  if (!email || !email.includes("@")) return "";
  return email.split("@").pop().toLowerCase();
}

export default function AuthPage({ mode = "login" }) {
  const navigate = useNavigate();
  const { login, register } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitError, setSubmitError] = useState("");
  const [loading, setLoading] = useState(false);

  // 关键：由路由传入的 mode 决定当前页面模式，避免状态错乱
  const isLogin = mode === "login";

  // Domain rules
  const emailDomain = useMemo(() => getEmailDomain(email), [email]);
  const isPublicDomain = !!emailDomain && PUBLIC_EMAIL_DOMAINS.has(emailDomain);
  const isEnterpriseDomain = emailDomain === ENTERPRISE_DOMAIN;

  // 仅注册时生效的限制
  const isRegisterBlocked = !isEnterpriseDomain || isPublicDomain;

  const inlineEmailHint = useMemo(() => {
    if (isLogin) return "";
    if (!email) return `仅支持企业邮箱：@${ENTERPRISE_DOMAIN}`;
    if (isPublicDomain) return "检测到公共邮箱域名，请改用企业邮箱。";
    if (!isEnterpriseDomain) return `邮箱域名需为 @${ENTERPRISE_DOMAIN}`;
    return "企业邮箱验证通过。";
  }, [email, isEnterpriseDomain, isPublicDomain, isLogin]);

  const inlineEmailHintTone = useMemo(() => {
    if (isLogin) return "text-slate-500";
    if (!email) return "text-slate-500";
    if (isPublicDomain) return "text-red-300";
    if (!isEnterpriseDomain) return "text-amber-300";
    return "text-emerald-300";
  }, [email, isEnterpriseDomain, isPublicDomain, isLogin]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitError("");
    setLoading(true);

    try {
      if (isLogin) {
        await login(email, password);
      } else {
        if (isRegisterBlocked) {
          throw new Error(`仅支持 ${ENTERPRISE_DOMAIN} 企业邮箱注册`);
        }
        await register(email, password);
      }

      // 兼容：你的 router 如果不支持 replace options，也能正常跳
      try {
        navigate("/app", { replace: true });
      } catch {
        navigate("/app");
      }
    } catch (err) {
      // 建议保留 console，方便定位后端返回
      // eslint-disable-next-line no-console
      console.error("AUTH_ERROR:", err);
      setSubmitError(
        err?.response?.data?.message ||
          err?.message ||
          "操作失败，请检查网络或凭据"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen w-full bg-[#020617] text-slate-200 overflow-hidden font-sans selection:bg-purple-500/30">
      {/* Background Layer */}
      <div className="absolute inset-0 pointer-events-none">
        <div className={`absolute inset-0 ${gradientBg}`} />
        <div className="absolute inset-0 opacity-[0.03] bg-[url('https://grainy-gradients.vercel.app/noise.svg')]" />
        {/* Animated Orbs */}
        <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-purple-600/10 blur-[120px] rounded-full animate-pulse" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-cyan-500/10 blur-[120px] rounded-full animate-pulse [animation-delay:2s]" />
      </div>

      <div className="relative z-10 flex min-h-screen">
        {/* LEFT SIDE: AUTH PANEL */}
        <div className="w-full lg:w-[500px] xl:w-[580px] flex flex-col p-8 md:p-12 lg:p-16 border-r border-slate-800/40 bg-slate-950/20 backdrop-blur-md">
          {/* Brand Header */}
          <div className="mb-12 flex items-center gap-3 group cursor-default">
            <div className="relative">
              <div className="absolute inset-0 bg-yellow-400 blur-lg opacity-20 group-hover:opacity-40 transition-opacity" />
              <div className="relative bg-gradient-to-br from-yellow-400 to-amber-600 p-2.5 rounded-2xl shadow-xl shadow-yellow-900/20">
                <Zap className="text-slate-900 w-6 h-6 fill-current" />
              </div>
            </div>
            <div className="min-w-0">
              <h1 className="text-xl font-bold tracking-tight text-white">
                BananaFlow
              </h1>
              <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500 font-semibold">
                Intelligence Workbench
              </p>
            </div>

            <div className="ml-auto text-[11px] px-2.5 py-1 rounded-full border border-white/10 bg-white/5 text-slate-200/80">
              @{ENTERPRISE_DOMAIN}
            </div>
          </div>

          <div className="flex-1 flex flex-col justify-center max-w-md mx-auto w-full">
            {/* Header Text */}
            <div className="mb-10 space-y-2">
              <h2 className="text-3xl font-bold text-white tracking-tight">
                {isLogin ? "欢迎回来" : "创建企业账户"}
              </h2>
              <p className="text-slate-400 text-sm">
                {isLogin
                  ? "登录后进入工作台"
                  : "注册后自动登录，进入工作台开始创作"}
              </p>
            </div>

            {/* Auth Form */}
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-4">
                {/* Email */}
                <div className="space-y-2">
                  <label className="text-xs font-medium text-slate-400 ml-1">
                    企业邮箱
                  </label>

                  <div className="relative group">
                    <Mail className="w-4 h-4 text-slate-500 absolute left-4 top-1/2 -translate-y-1/2 group-focus-within:text-purple-400 transition-colors" />
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      placeholder={`yourname@${ENTERPRISE_DOMAIN}`}
                      className={`${inputStyle} pl-11`}
                      autoComplete="email"
                    />
                  </div>

                  {/* 注册模式提示（始终显示一行，不会“没提示就按钮灰掉”） */}
                  {!isLogin && (
                    <div
                      className={`text-[11px] px-1 flex items-center gap-1.5 ${inlineEmailHintTone}`}
                    >
                      {isEnterpriseDomain ? (
                        <CheckCircle2 className="w-3 h-3" />
                      ) : (
                        <Shield className="w-3 h-3" />
                      )}
                      <span>{inlineEmailHint}</span>
                    </div>
                  )}
                </div>

                {/* Password */}
                <div className="space-y-2">
                  <label className="text-xs font-medium text-slate-400 ml-1">
                    密码
                  </label>
                  <div className="relative group">
                    <Lock className="w-4 h-4 text-slate-500 absolute left-4 top-1/2 -translate-y-1/2 group-focus-within:text-purple-400 transition-colors" />
                    <input
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      minLength={6}
                      required
                      placeholder="至少 6 位"
                      className={`${inputStyle} pl-11`}
                      autoComplete={isLogin ? "current-password" : "new-password"}
                    />
                  </div>

                  <div className="text-[11px] text-slate-500 px-1">
                    建议使用公司统一密码规范（至少 6 位）
                  </div>
                </div>
              </div>

              {submitError && (
                <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-300 text-xs flex items-start gap-2">
                  <div className="mt-0.5">⚠️</div>
                  <span>{submitError}</span>
                </div>
              )}

              <button
                type="submit"
                // 关键：只在注册模式且不符合企业域时禁用
                disabled={loading || (!isLogin && isRegisterBlocked)}
                className={[
                  "w-full group relative overflow-hidden py-3.5 rounded-xl bg-white text-slate-950 font-bold text-sm",
                  "transition-all hover:scale-[1.02] active:scale-[0.98]",
                  "disabled:opacity-40 disabled:hover:scale-100",
                  "shadow-xl shadow-white/5",
                ].join(" ")}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-purple-200 to-white opacity-0 group-hover:opacity-100 transition-opacity" />
                <span className="relative flex items-center justify-center gap-2">
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      处理中...
                    </>
                  ) : (
                    <>{isLogin ? "登录账户" : "注册并开启"}</>
                  )}
                </span>
              </button>
            </form>

            {/* Switch link (route-driven) */}
            <div className="mt-8 text-center">
              <p className="text-sm text-slate-500">
                {isLogin ? "还没有账号？" : "已经拥有账号？"}
                <Link
                  to={isLogin ? "/register" : "/login"}
                  className="ml-2 text-purple-400 font-semibold hover:text-purple-300 transition-colors underline-offset-4 hover:underline"
                >
                  {isLogin ? "立即注册" : "去登录"}
                </Link>
              </p>
            </div>
          </div>

          {/* Footer */}
          <div className="mt-auto pt-8 flex items-center justify-between text-[10px] text-slate-600 uppercase tracking-widest font-bold">
            <span>© 2024 Dayu Keji</span>
            <div className="flex gap-4">
              <a href="#" className="hover:text-slate-400 transition-colors">
                Privacy
              </a>
              <a href="#" className="hover:text-slate-400 transition-colors">
                Terms
              </a>
            </div>
          </div>
        </div>

        {/* RIGHT SIDE: PREVIEW */}
        <div className="hidden lg:flex flex-1 relative items-center justify-center p-12 xl:p-24 overflow-hidden">
          <div className="absolute inset-0 opacity-20 [mask-image:radial-gradient(ellipse_at_center,black,transparent)] bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:40px_40px]" />

          <div className="relative w-full max-w-4xl aspect-[1.4] animate-in fade-in zoom-in duration-1000">
            <div className="absolute -inset-4 rounded-[3rem] bg-gradient-to-br from-white/10 to-white/0 border border-white/5 blur-xl" />
            <LoginFlowDemo />
          </div>
        </div>
      </div>
    </div>
  );
}
