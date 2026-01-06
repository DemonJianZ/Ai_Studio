import React, { Children, createContext, isValidElement, useCallback, useContext, useEffect, useMemo, useState } from "react";

const RouterContext = createContext(null);

const normalizePath = (path) => {
  if (!path) return "/";
  if (path.startsWith("http")) return path;
  return path.startsWith("/") ? path : `/${path}`;
};

const matchPath = (routePath, currentPath) => {
  if (routePath === "*") return true;
  return normalizePath(routePath) === normalizePath(currentPath);
};

export function BrowserRouter({ children }) {
  const [path, setPath] = useState(() => window.location.pathname);

  const navigate = useCallback((to, options = {}) => {
    const target = normalizePath(to);
    if (options.replace) {
      window.history.replaceState({}, "", target);
    } else {
      window.history.pushState({}, "", target);
    }
    setPath(target);
  }, []);

  useEffect(() => {
    const handler = () => setPath(window.location.pathname);
    window.addEventListener("popstate", handler);
    return () => window.removeEventListener("popstate", handler);
  }, []);

  const value = useMemo(() => ({ path, navigate }), [path, navigate]);

  return <RouterContext.Provider value={value}>{children}</RouterContext.Provider>;
}

export function Routes({ children }) {
  const { path } = useContext(RouterContext);
  let element = null;

  Children.forEach(children, (child) => {
    if (element || !isValidElement(child)) return;
    if (matchPath(child.props.path, path)) {
      element = child.props.element;
    }
  });

  return element;
}

export function Route() {
  return null;
}

export function Navigate({ to, replace }) {
  const navigate = useNavigate();
  useEffect(() => {
    navigate(to, { replace });
  }, [navigate, replace, to]);
  return null;
}

export function useNavigate() {
  const ctx = useContext(RouterContext);
  if (!ctx) throw new Error("useNavigate must be used within BrowserRouter");
  return ctx.navigate;
}

export function Link({ to, children, className }) {
  const navigate = useNavigate();
  const handleClick = (e) => {
    e.preventDefault();
    navigate(to);
  };
  return (
    <a href={normalizePath(to)} onClick={handleClick} className={className}>
      {children}
    </a>
  );
}
