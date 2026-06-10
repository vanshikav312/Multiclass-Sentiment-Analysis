import React, { useState } from 'react';
import { Sparkles, Brain, AlertCircle, RefreshCw } from 'lucide-react';

export default function Analyzer() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) {
        throw new Error('Analysis failed. Please check the backend connection.');
      }
      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getEmotionColor = (emotion) => {
    const map = {
      joy: 'var(--color-orange)',
      happy: 'var(--color-orange)',
      sadness: '#60a5fa',
      sad: '#60a5fa',
      anger: '#f87171',
      fear: '#fbbf24',
      surprise: '#a78bfa',
      disgust: '#a16207',
      neutral: '#94a3b8'
    };
    return map[emotion.toLowerCase()] || 'var(--color-burnt-orange)';
  };

  // Helper to render progress bar
  const renderProgressBar = (value, color) => {
    const pct = Math.round(value * 100);
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', width: '100%', marginBottom: '8px' }}>
        <div style={{ flex: 1, height: '8px', background: 'rgba(255,255,255,0.06)', borderRadius: '99px', overflow: 'hidden' }}>
          <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: '99px', transition: 'width 0.8s ease' }} />
        </div>
        <span style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-primary)', width: '36px', textAlign: 'right' }}>{pct}%</span>
      </div>
    );
  };

  return (
    <div className="animate-fade-in" style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '32px' }}>
      
      {/* Header section */}
      <div>
        <h1 style={{ fontSize: '36px', fontWeight: '800', marginBottom: '8px' }}>
          <span className="gradient-text">MindScan</span> Sentiment Analyzer
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '16px' }}>
          Enter a paragraph, journal entry, or live sentence to perform multi-class mental health sentiment analysis.
        </p>
      </div>

      {/* Input panel */}
      <div className="glass-panel" style={{ padding: '28px', position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '4px', background: 'var(--gradient-3)' }} />
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. I've been feeling really overwhelmed with everything happening at work lately. It feels like I'm running out of time and my chest gets tight when I think about tomorrow..."
          style={{
            width: '100%',
            height: '140px',
            background: 'rgba(16, 12, 8, 0.4)',
            border: '1px solid rgba(202, 63, 22, 0.2)',
            borderRadius: '12px',
            padding: '16px',
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-body)',
            fontSize: '15px',
            lineHeight: '1.6',
            resize: 'none',
            outline: 'none',
            transition: 'var(--transition-fast)',
            marginBottom: '20px'
          }}
          onFocus={(e) => e.target.style.borderColor = 'var(--color-orange)'}
          onBlur={(e) => e.target.style.borderColor = 'rgba(202, 63, 22, 0.2)'}
        />
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Characters: {text.length} | Words: {text.trim() ? text.trim().split(/\s+/).length : 0}
          </span>
          <button
            onClick={handleAnalyze}
            disabled={loading || !text.trim()}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '12px 28px',
              borderRadius: '12px',
              border: 'none',
              background: text.trim() ? 'var(--gradient-3)' : 'rgba(149, 18, 44, 0.2)',
              color: text.trim() ? '#fff' : 'var(--text-muted)',
              cursor: text.trim() && !loading ? 'pointer' : 'not-allowed',
              fontWeight: '600',
              fontFamily: 'var(--font-display)',
              fontSize: '15px',
              boxShadow: text.trim() ? '0 4px 20px rgba(255, 148, 8, 0.25)' : 'none',
              transition: 'var(--transition-fast)'
            }}
            onMouseEnter={(e) => {
              if (text.trim() && !loading) {
                e.currentTarget.style.opacity = '0.9';
                e.currentTarget.style.transform = 'translateY(-1px)';
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.opacity = '1';
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            {loading ? (
              <>
                <RefreshCw size={18} className="animate-spin" style={{ animation: 'spin 1.5s linear infinite' }} />
                Analyzing...
              </>
            ) : (
              <>
                <Sparkles size={18} />
                Analyze Sentiment
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="glass-panel" style={{ padding: '16px 20px', borderColor: '#ef4444', display: 'flex', alignItems: 'center', gap: '12px', background: 'rgba(239, 68, 68, 0.05)' }}>
          <AlertCircle color="#ef4444" size={20} />
          <span style={{ color: '#fca5a5', fontSize: '14px', fontWeight: '500' }}>{error}</span>
        </div>
      )}

      {/* Results panel */}
      {results && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '32px' }}>
          
          {/* Emotion Classification card */}
          <div className="glass-panel" style={{ padding: '28px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div>
              <span style={{ fontSize: '12px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '1.5px', color: 'var(--color-orange)' }}>Primary Emotion</span>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', marginTop: '4px' }}>
                <h2 style={{ fontSize: '42px', fontWeight: '800', textTransform: 'capitalize', color: 'var(--text-primary)' }}>
                  {Object.keys(results.emotion).reduce((a, b) => results.emotion[a] > results.emotion[b] ? a : b)}
                </h2>
                <span style={{
                  fontSize: '14px',
                  fontWeight: '600',
                  color: 'var(--color-orange)',
                  background: 'rgba(255, 148, 8, 0.1)',
                  padding: '4px 12px',
                  borderRadius: '100px'
                }}>
                  {Math.round(Math.max(...Object.values(results.emotion)) * 100)}% Confidence
                </span>
              </div>
            </div>

            <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)', paddingPrompt: '16px' }} />

            <div>
              <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px', color: 'var(--text-secondary)' }}>Emotion Probabilities</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                {Object.entries(results.emotion)
                  .sort((a, b) => b[1] - a[1])
                  .map(([emotion, value]) => (
                    <div key={emotion}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <span style={{ textTransform: 'capitalize', fontSize: '14px', fontWeight: '600', color: 'var(--text-secondary)' }}>{emotion}</span>
                      </div>
                      {renderProgressBar(value, getEmotionColor(emotion))}
                    </div>
                  ))}
              </div>
            </div>
          </div>

          {/* Secondary Classifier cards */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            
            {/* Depression Card */}
            <div className="glass-panel" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <Brain size={18} color="var(--color-orange)" />
                  <h3 style={{ fontSize: '16px', fontWeight: '700' }}>Depression Risk</h3>
                </div>
                <span style={{
                  fontSize: '12px',
                  fontWeight: '700',
                  color: results.depression.Depressed > 0.5 ? '#ef4444' : '#10b981',
                  background: results.depression.Depressed > 0.5 ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                  padding: '4px 10px',
                  borderRadius: '100px',
                  textTransform: 'uppercase'
                }}>
                  {results.depression.Depressed > 0.5 ? 'High Risk' : 'Low Risk'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                <span>Depressed Probability</span>
                <span>{Math.round(results.depression.Depressed * 100)}%</span>
              </div>
              {renderProgressBar(results.depression.Depressed, results.depression.Depressed > 0.5 ? 'var(--color-crimson)' : '#10b981')}
            </div>

            {/* Anxiety Card */}
            <div className="glass-panel" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <Brain size={18} color="var(--color-orange)" />
                  <h3 style={{ fontSize: '16px', fontWeight: '700' }}>Anxiety Risk</h3>
                </div>
                <span style={{
                  fontSize: '12px',
                  fontWeight: '700',
                  color: results.anxiety.Anxious > 0.5 ? '#ef4444' : '#10b981',
                  background: results.anxiety.Anxious > 0.5 ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                  padding: '4px 10px',
                  borderRadius: '100px',
                  textTransform: 'uppercase'
                }}>
                  {results.anxiety.Anxious > 0.5 ? 'High Risk' : 'Low Risk'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                <span>Anxious Probability</span>
                <span>{Math.round(results.anxiety.Anxious * 100)}%</span>
              </div>
              {renderProgressBar(results.anxiety.Anxious, results.anxiety.Anxious > 0.5 ? 'var(--color-burnt-orange)' : '#10b981')}
            </div>

            {/* Eating Behavior Card */}
            <div className="glass-panel" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
                <Brain size={18} color="var(--color-orange)" />
                <h3 style={{ fontSize: '16px', fontWeight: '700' }}>Eating Behavior</h3>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {Object.entries(results.eating)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 3)
                  .map(([behavior, value]) => (
                    <div key={behavior} style={{ marginBottom: '4px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '2px' }}>
                        <span style={{ textTransform: 'capitalize' }}>{behavior}</span>
                        <span>{Math.round(value * 100)}%</span>
                      </div>
                      {renderProgressBar(value, 'var(--gradient-1)')}
                    </div>
                  ))}
              </div>
            </div>

          </div>

        </div>
      )}

    </div>
  );
}
