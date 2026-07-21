"use client";

import { FormEvent, useState } from "react";

type AnalysisResponse = {
  id: string;
  status: "queued" | "processing" | "completed" | "failed";
  result?: {
    score: number;
    feedback: string;
    details: {
      detection_rate: number;
      frames_analyzed: number;
      total_frames: number;
      consistency: number;
    };
  } | null;
  error_message?: string | null;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const sleep = (milliseconds: number) => new Promise((resolve) => setTimeout(resolve, milliseconds));

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function poll(videoId: string): Promise<AnalysisResponse> {
    for (;;) {
      const response = await fetch(`${API_URL}/api/videos/${videoId}`, { cache: "no-store" });
      const payload: AnalysisResponse = await response.json();
      if (!response.ok) throw new Error("Could not retrieve analysis status");
      setAnalysis(payload);
      if (payload.status === "completed") return payload;
      if (payload.status === "failed") throw new Error(payload.error_message ?? "Analysis failed");
      await sleep(1500);
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;

    setIsSubmitting(true);
    setError(null);
    setAnalysis(null);
    const body = new FormData();
    body.append("video", file);

    try {
      const response = await fetch(`${API_URL}/api/videos`, { method: "POST", body });
      const payload: AnalysisResponse & { detail?: string } = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Analysis failed");
      setAnalysis(payload);
      await poll(payload.id);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Analysis failed");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main>
      <section className="card">
        <p className="eyebrow">SquatSpot</p>
        <h1>Analyze your squat</h1>
        <p className="lede">
          The uploaded video is used only for pose extraction and deleted after processing.
        </p>

        <form onSubmit={submit}>
          <label className="upload">
            <span>{file ? file.name : "Choose an MP4, AVI, MOV, or MKV video"}</span>
            <input
              type="file"
              accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <button disabled={!file || isSubmitting} type="submit">
            {isSubmitting ? "Analyzing…" : "Analyze video"}
          </button>
        </form>

        {analysis && !analysis.result && !error && (
          <p>Analysis status: {analysis.status}</p>
        )}
        {error && <p className="error">{error}</p>}

        {analysis?.result && (
          <article className="result">
            <div>
              <span className="score">{analysis.result.score}</span>
              <span>/100</span>
            </div>
            <p>{analysis.result.feedback}</p>
            <dl>
              <div><dt>Detection</dt><dd>{analysis.result.details.detection_rate}%</dd></div>
              <div><dt>Consistency</dt><dd>{analysis.result.details.consistency}%</dd></div>
              <div><dt>Frames analyzed</dt><dd>{analysis.result.details.frames_analyzed}</dd></div>
            </dl>
          </article>
        )}
      </section>
    </main>
  );
}
