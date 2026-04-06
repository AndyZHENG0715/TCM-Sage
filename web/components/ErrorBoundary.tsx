"use client";

import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
    children: ReactNode;
}

interface ErrorBoundaryState {
    hasError: boolean;
    error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
    state: ErrorBoundaryState = {
        hasError: false,
        error: null,
    };

    static getDerivedStateFromError(error: Error): ErrorBoundaryState {
        return {
            hasError: true,
            error,
        };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error("ErrorBoundary caught an error:", error, errorInfo);
    }

    handleRetry = () => {
        this.setState({ hasError: false, error: null });
    };

    render() {
        if (this.state.hasError) {
            return (
                <div className="flex min-h-[320px] w-full items-center justify-center rounded-2xl border border-primary/30 bg-background-dark px-6 py-10 text-parchment">
                    <div className="max-w-md space-y-4 text-center">
                        <div className="space-y-2">
                            <h2 className="text-xl font-semibold text-primary">Something went wrong / 页面发生错误</h2>
                            <p className="text-sm text-parchment/80">We couldn&apos;t render this section. Please try again. / 当前区域无法显示，请重试。</p>
                        </div>
                        {this.state.error?.message ? (
                            <div className="rounded-lg border border-primary/20 bg-black/20 px-4 py-3 text-left text-xs text-parchment/70">
                                {this.state.error.message}
                            </div>
                        ) : null}
                        <button
                            type="button"
                            onClick={this.handleRetry}
                            className="inline-flex min-h-11 items-center justify-center rounded-lg border border-primary bg-primary px-4 py-2 text-sm font-semibold text-background-dark transition-colors hover:bg-primary/90"
                        >
                            Retry / 重试
                        </button>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}
