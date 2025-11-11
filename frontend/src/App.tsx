import { useCallback, useEffect, useMemo, useState } from "react";

type Role = "user" | "assistant" | "system";

interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  timestamp: string;
}

interface AnalysisJob {
  job_id: string;
  status: string;
  summary?: string | null;
  report_path?: string | null;
  error?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
  result?: Record<string, unknown> | null;
}

const defaultBaseUrl =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ||
  "/api";

const POLL_INTERVAL_MS = 3000;

function makeId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

export default function App() {
  const [company, setCompany] = useState("");
  const [drug, setDrug] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [pollingJobId, setPollingJobId] = useState<string | null>(null);
  const [activeJobStatus, setActiveJobStatus] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [completedJobId, setCompletedJobId] = useState<string | null>(null);
  const [reportQuestion, setReportQuestion] = useState("");
  const [isQueryingReport, setIsQueryingReport] = useState(false);

  const apiBaseUrl = useMemo(() => defaultBaseUrl, []);

  const downloadReport = useCallback(
    async (jobId: string, reportPath: string | null | undefined) => {
      if (!reportPath) {
        throw new Error("Analysis completed but no report is available for download.");
      }

      const response = await fetch(`${apiBaseUrl}/analyze/${jobId}/report`);
      if (!response.ok) {
        throw new Error(`Failed to download report (${response.status})`);
      }

      const blob = await response.blob();
      const filename =
        reportPath.split(/[\\/]/).filter(Boolean).pop() ?? `analysis-${jobId}.pdf`;
      const url = window.URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.URL.revokeObjectURL(url);
    },
    [apiBaseUrl],
  );

  const appendMessage = useCallback((role: Role, content: string) => {
    setMessages((prev) => [
      ...prev,
      {
        id: makeId(),
        role,
        content,
        timestamp: new Date().toISOString(),
      },
    ]);
  }, []);

  const pollJob = useCallback(
    async (jobId: string) => {
      try {
        const response = await fetch(`${apiBaseUrl}/analyze/${jobId}`);
        if (!response.ok) {
          throw new Error(`Job status request failed (${response.status})`);
        }

        const job: AnalysisJob = await response.json();
        setActiveJobStatus(job.status);

        if (job.status === "completed") {
          try {
            await downloadReport(jobId, job.report_path);
          } catch (downloadError) {
            console.error(downloadError);
            setErrorMessage((downloadError as Error).message);
          }
          setPollingJobId(null);
          setIsSubmitting(false);
          setCompletedJobId(jobId);
        } else if (job.status === "failed") {
          const reason = job.error?.trim() || "Unknown error";
          appendMessage("system", `Analysis failed: ${reason}`);
          setPollingJobId(null);
          setIsSubmitting(false);
          setCompletedJobId(null);
        }
      } catch (error) {
        console.error(error);
        setErrorMessage((error as Error).message);
        setPollingJobId(null);
        setIsSubmitting(false);
        setCompletedJobId(null);
      }
    },
    [apiBaseUrl, appendMessage, downloadReport],
  );

  useEffect(() => {
    if (!pollingJobId) {
      return;
    }

    let cancelled = false;

    const runPoll = async () => {
      if (!cancelled) {
        await pollJob(pollingJobId);
      }
    };

    void runPoll();
    const interval = window.setInterval(runPoll, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [pollingJobId, pollJob]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setErrorMessage(null);
    setCompletedJobId(null);

    if (!company.trim() || !drug.trim()) {
      setErrorMessage("Company and drug are required.");
      return;
    }

    appendMessage(
      "user",
      `Please analyze ${drug.trim()} from ${company.trim()}.`,
    );

    setIsSubmitting(true);
    setActiveJobStatus("pending");

    try {
      const response = await fetch(`${apiBaseUrl}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          company: company.trim(),
          drug: drug.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to queue analysis (${response.status})`);
      }

      const job: AnalysisJob = await response.json();
      setPollingJobId(job.job_id);
      setActiveJobStatus(job.status);
      setCompletedJobId(null);
    } catch (error) {
      console.error(error);
      setErrorMessage((error as Error).message);
      setIsSubmitting(false);
    }
  };

  const handleReportQuerySubmit = async (
    event: React.FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault();
    setErrorMessage(null);

    if (!completedJobId) {
      setErrorMessage("No completed analysis available to query.");
      return;
    }

    const trimmedQuestion = reportQuestion.trim();
    if (!trimmedQuestion) {
      return;
    }

    appendMessage("user", trimmedQuestion);
    setIsQueryingReport(true);
    setReportQuestion("");

    try {
      const response = await fetch(
        `${apiBaseUrl}/analyze/${completedJobId}/query`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: trimmedQuestion }),
        },
      );

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        const detail =
          (data && (data.detail as string | undefined)) ||
          `Failed to query report (${response.status})`;
        throw new Error(detail);
      }

      const { answer } = (await response.json()) as { answer: string };
      appendMessage("assistant", answer.trim());
    } catch (error) {
      console.error(error);
      setErrorMessage((error as Error).message);
    } finally {
      setIsQueryingReport(false);
    }
  };

  return (
    <div className="page">
      <header className="header">
        <h1>PharmaSight</h1>
        <p>Enter a company and drug to generate a comprehensive analysis of the drug's clinical trial success probability.</p>
      </header>

      <main className="content">
        <section className="form-section">
          <form className="input-form" onSubmit={handleSubmit}>
            <label>
              Company
              <input
                value={company}
                onChange={(event) => setCompany(event.target.value)}
                placeholder="Company name"
                required
              />
            </label>
            <label>
              Drug
              <input
                value={drug}
                onChange={(event) => setDrug(event.target.value)}
                placeholder="Drug name"
                required
              />
            </label>
            <button type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Running analysis..." : "Start analysis"}
            </button>
            {activeJobStatus && (
              <span className={`status status-${activeJobStatus}`}>
                Status: {activeJobStatus}
              </span>
            )}
            {errorMessage && <span className="error">{errorMessage}</span>}
          </form>
        </section>

        <section className="chat-section">
          <h2>Chat Transcript</h2>
          <div className="chat-log">
            {messages.length === 0 && (
              <p className="placeholder">
                Submit the form to start a new conversation with the analysis
                agents.
              </p>
            )}
            {messages.map((message) => (
              <article key={message.id} className={`message ${message.role}`}>
                <header>
                  <strong>{message.role.toUpperCase()}</strong>
                  <time dateTime={message.timestamp}>
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </time>
                </header>
                <p>{message.content}</p>
              </article>
            ))}
          </div>
          <form className="chat-query-form" onSubmit={handleReportQuerySubmit}>
            <input
              type="text"
              value={reportQuestion}
              onChange={(event) => setReportQuestion(event.target.value)}
              placeholder={
                completedJobId
                  ? "Ask a question about the most recent report..."
                  : "Run an analysis to enable report Q&A."
              }
              disabled={!completedJobId || isQueryingReport}
            />
            <button type="submit" disabled={!completedJobId || isQueryingReport}>
              {isQueryingReport ? "Thinking..." : "Ask"}
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}

