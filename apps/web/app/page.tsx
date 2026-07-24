"use client";

import { FormEvent, useState } from "react";

type ErrorEvent = {
  id: string;
  type: "bad_back" | "bad_heel";
  start_timestamp_ms: number;
  end_timestamp_ms: number;
  duration_ms: number;
  evidence_frame_count: number;
  confidence: number;
  representative_frame_number: number;
  representative_timestamp_ms: number;
  frame_image_url: string | null;
  measurements: Record<string, number | null>;
  artifact_warning?: string | null;
};

type AnalysisResponse = {
  id: string;
  status: "queued" | "processing" | "completed" | "failed";
  extraction?: {
    schema_version: number;
    model: string;
    video: {
      fps: number;
      total_frames: number;
      duration_seconds: number;
      width: number;
      height: number;
      content_type: string;
    };
    sampling_fps: number;
    sampled_frame_count: number;
    detected_frame_count: number;
    detection_rate: number;
  } | null;
  classification?: {
    result: "good" | "errors_detected" | "insufficient_pose_data";
    event_count: number;
    event_counts: Record<string, number>;
    events: ErrorEvent[];
    classified_frame_count: number;
    sampled_frame_count: number;
    classified_frame_coverage: number;
  } | null;
  progress?: {
    decoded_frame_count: number;
    sampled_frame_count: number;
    detected_frame_count: number;
    classified_frame_count: number;
    total_frame_count: number;
    percent: number;
  } | null;
  error_message?: string | null;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const sleep = (milliseconds: number) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));
const measurementLabel = (name: string) =>
  name.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function poll(videoId: string): Promise<AnalysisResponse> {
    for (;;) {
      const response = await fetch(`${API_URL}/api/videos/${videoId}`, {
        cache: "no-store",
      });
      const payload: AnalysisResponse = await response.json();
      if (!response.ok) throw new Error("Could not retrieve analysis status");
      setAnalysis(payload);
      if (payload.status === "completed") return payload;
      if (payload.status === "failed") {
        throw new Error(payload.error_message ?? "Analysis failed");
      }
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
      const response = await fetch(`${API_URL}/api/videos`, {
        method: "POST",
        body,
      });
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

  async function deleteAnalysis() {
    if (!analysis) return;
    const response = await fetch(`${API_URL}/api/videos/${analysis.id}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      setError("Could not delete this analysis");
      return;
    }
    setAnalysis(null);
    setFile(null);
  }

  return (
    <main>
      <section className="card">
        <p className="eyebrow">SquatSpot</p>
        <h1>Analyze your squat</h1>
        <p className="lede">
          Your video is analyzed locally and deleted after processing.
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

        {analysis && !analysis.extraction && !error && (
          <section className="progress-panel" aria-live="polite">
            <p>Analysis status: {analysis.status}</p>
            {analysis.progress && (
              <>
                <progress max="100" value={analysis.progress.percent} />
                <p>
                  {analysis.progress.percent.toFixed(1)}% · sampled{" "}
                  {analysis.progress.sampled_frame_count} frames · detected{" "}
                  {analysis.progress.detected_frame_count} poses
                </p>
              </>
            )}
          </section>
        )}
        {error && <p className="error">{error}</p>}

        {analysis?.extraction && (
          <article className="result">
            <h2>
              {analysis.classification?.result === "good"
                ? "Good squat form"
                : analysis.classification?.result === "errors_detected"
                  ? "Form errors detected"
                  : analysis.classification?.result === "insufficient_pose_data"
                    ? "Not enough pose data to assess form"
                  : "Landmark extraction complete"}
            </h2>

            {analysis.classification?.result === "insufficient_pose_data" && (
              <p>
                Too few sampled frames contained enough reliable biomechanical
                measurements. Try a clearer full-body view with the hips, knees,
                ankles, and feet visible.
              </p>
            )}

            {analysis.classification?.events.length ? (
              <div className="event-list">
                {analysis.classification.events.map((detectedEvent) => (
                  <section className="event-card" key={detectedEvent.id}>
                    {detectedEvent.frame_image_url ? (
                      <img
                        alt={`Representative ${detectedEvent.type.replace("_", " ")} frame`}
                        loading="lazy"
                        src={`${API_URL}${detectedEvent.frame_image_url}`}
                      />
                    ) : (
                      <div className="image-fallback">Representative frame unavailable</div>
                    )}
                    <div className="event-body">
                      <h3>
                        {detectedEvent.type === "bad_back"
                          ? "Back position"
                          : "Heel position"}
                      </h3>
                      <p>
                        {(detectedEvent.confidence * 100).toFixed(1)}% confidence ·{" "}
                        {(detectedEvent.start_timestamp_ms / 1000).toFixed(1)}–
                        {(detectedEvent.end_timestamp_ms / 1000).toFixed(1)}s ·{" "}
                        {detectedEvent.evidence_frame_count} evidence frames
                      </p>
                      <dl className="measurements">
                        {Object.entries(detectedEvent.measurements)
                          .filter((entry): entry is [string, number] =>
                            Number.isFinite(entry[1])
                          )
                          .slice(0, 6)
                          .map(([name, value]) => (
                            <div key={name}>
                              <dt>{measurementLabel(name)}</dt>
                              <dd>{value.toFixed(1)}</dd>
                            </div>
                          ))}
                      </dl>
                    </div>
                  </section>
                ))}
              </div>
            ) : null}

            {analysis.classification && (
              <p>
                Classified {analysis.classification.classified_frame_count} frames (
                {analysis.classification.classified_frame_coverage}% coverage).
              </p>
            )}
            <p>
              MediaPipe detected a pose in {analysis.extraction.detected_frame_count} of{" "}
              {analysis.extraction.sampled_frame_count} sampled frames.
            </p>
            <dl>
              <div><dt>Detection rate</dt><dd>{analysis.extraction.detection_rate}%</dd></div>
              <div><dt>Sampling rate</dt><dd>{analysis.extraction.sampling_fps} FPS</dd></div>
              <div><dt>Duration</dt><dd>{analysis.extraction.video.duration_seconds.toFixed(1)}s</dd></div>
              <div><dt>Resolution</dt><dd>{analysis.extraction.video.width} × {analysis.extraction.video.height}</dd></div>
              <div><dt>Model</dt><dd>{analysis.extraction.model}</dd></div>
              <div><dt>Status</dt><dd>Completed</dd></div>
            </dl>
            {analysis.classification && (
              <small>
                This result comes from a lightweight model running locally and is not
                medical advice.
              </small>
            )}
            <button className="delete-button" onClick={deleteAnalysis} type="button">
              Delete analysis
            </button>
          </article>
        )}
      </section>
    </main>
  );
}
