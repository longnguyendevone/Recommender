import React from "react";

function ScoreBar({ label, value, colorClass }) {
  const pct = Math.max(0, Math.min(100, Math.round((value || 0) * 100)));
  return (
    <div className="score-bar-row">
      <span className="score-bar-label">{label}</span>
      <div className="score-bar-track">
        <div
          className={`score-bar-fill ${colorClass}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="score-bar-val">{(value || 0).toFixed(2)}</span>
    </div>
  );
}

export default function RecommendationList({
  responseMeta,
  recommendations,
  selectedRec,
  onSelectRec,
}) {
  return (
    <div className="rec-panel">
      <div className="rec-header">
        <h2>Top-K Recommendations</h2>
        {responseMeta && (
          <div className="meta-badges">
            <span className="badge badge-model">{responseMeta.model_used}</span>
            <span className="badge badge-time">{responseMeta.inference_time_ms} ms</span>
            {responseMeta.is_cold_start && (
              <span className="badge badge-cold">Cold-Start</span>
            )}
          </div>
        )}
      </div>

      {recommendations.length === 0 ? (
        <div className="empty-state">
          No venues match the current radius or category constraints. Try widening the radius or clearing category filters.
        </div>
      ) : (
        <div className="rec-cards">
          {recommendations.map((rec) => {
            const isSelected = selectedRec && selectedRec.poi_id === rec.poi_id;
            return (
              <button
                type="button"
                key={`${rec.poi_id}-${rec.rank}`}
                className={isSelected ? "rec-card selected" : "rec-card"}
                onClick={() => onSelectRec(rec)}
              >
                <div className="rec-card-top">
                  <span className="rec-rank">#{rec.rank}</span>
                  <div className="rec-title-wrap">
                    <div className="rec-title">{rec.name}</div>
                    <div className="rec-sub">
                      <span className="cat-pill">{rec.category}</span>
                      <span>{rec.distance_km.toFixed(2)} km</span>
                      <span>Score {rec.total_score.toFixed(3)}</span>
                    </div>
                  </div>
                </div>

                <div className="rec-explanation">{rec.explanation}</div>

                <div className="score-bars">
                  <ScoreBar label="Seq" value={rec.score_breakdown.sequential} colorClass="fill-seq" />
                  <ScoreBar label="Cnt" value={rec.score_breakdown.content} colorClass="fill-cnt" />
                  <ScoreBar label="CF" value={rec.score_breakdown.cf} colorClass="fill-cf" />
                  <ScoreBar label="Geo" value={rec.score_breakdown.spatial} colorClass="fill-geo" />
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
