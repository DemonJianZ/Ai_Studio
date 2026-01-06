import React from "react";
import { BrowserRouter, Navigate, Route, Routes } from "./router";
import { AuthProvider, useAuth } from "./auth/AuthProvider";
import Workbench from "./pages/Workbench";
import AuthPage from "./pages/AuthPage";

const LoadingScreen = () => (
  <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center">
    <div className="flex flex-col items-center gap-2">
      <div className="h-10 w-10 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
      <div className="text-sm text-slate-400">初始化中...</div>
    </div>
  </div>
);

const ProtectedRoute = ({ children }) => {
  const { user, token, loading } = useAuth();
  if (loading) return <LoadingScreen />;
  if (!token || !user) return <Navigate to="/login" replace />;
  return children;
};

const RootRedirect = () => {
  const { user, token, loading } = useAuth();
  if (loading) return <LoadingScreen />;
  if (token && user) return <Navigate to="/app" replace />;
  return <Navigate to="/login" replace />;
};

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<AuthPage mode="login" />} />
          <Route path="/register" element={<AuthPage mode="register" />} />
          <Route
            path="/app"
            element={
              <ProtectedRoute>
                <Workbench />
              </ProtectedRoute>
            }
          />
          <Route path="/" element={<RootRedirect />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
