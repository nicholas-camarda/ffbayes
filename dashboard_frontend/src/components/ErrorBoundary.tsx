import { Component, type ErrorInfo, type ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('Dashboard render error:', error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (error) {
      return (
        <div className="error-boundary" role="alert">
          <h1>FFBayes Draft War Room</h1>
          <p className="error-message">{error.message}</p>
          <p className="error-hint">
            The dashboard payload may be malformed — regenerate with{' '}
            <code>ffbayes stage-dashboard --year &lt;year&gt;</code>
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}
