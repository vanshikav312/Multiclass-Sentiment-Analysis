import React, { useState, useEffect } from 'react';
import { TrendingUp, FileText, CheckCircle, BarChart2, HelpCircle } from 'lucide-react';

export default function Dashboard() {
  const [metrics, setMetrics] = useState([]);
  const [explanations, setExplanations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('summary'); // summary, confusion, roc, distribution

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [mRes, eRes] = await Promise.all([
          fetch('http://127.0.0.1:8000/api/metrics'),
          fetch('http://127.0.0.1:8000/api/explain')
        ]);
        
        if (!mRes.ok || !eRes.ok) {
          throw new Error('Could not fetch evaluation metrics.');
        }

        const mData = await mRes.json();
        const eData = await eRes.json();

        setMetrics(mData);
        setExplanations(eData);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const getMetricColor = (val) => {
    const num = parseFloat(val);
    if (isNaN(num)) return 'var(--text-primary)';
    if (num >= 0.95) return '#10b981'; // Green for high performance
    if (num >= 0.90) return 'var(--color-orange)';
    if (num >= 0.70) return 'var(--color-burnt-orange)';
    return 'var(--color-crimson)';
  };

  const getExplanationKey = (modelName) => {
    // Maps model name in CSV to explanation dictionary keys
    if (modelName.toLowerCase().includes('depression')) return 'Depression';
    if (modelName.toLowerCase().includes('emotion')) return 'Emotion';
    if (modelName.toLowerCase().includes('eating')) return 'Eating Behavior';
    if (modelName.toLowerCase().includes('anxiety')) return 'Anxiety';
    return null;
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '300px' }}>
        <p style={{ color: 'var(--text-secondary)', fontSize: '16px' }}>Loading evaluation dashboard...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-panel" style={{ padding: '24px', borderColor: 'rgba(251,191,36,0.3)', color: 'var(--text-primary)', background: 'rgba(251,191,36,0.04)' }}>
        <h3 style={{ marginBottom: '8px', color: '#fde68a' }}>Loading models + data — takes 1–2 minutes on first run.</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Please wait a moment, then refresh the page.</p>
      </div>
    );
  }

  return (
    <div className="animate-fade-in" style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '32px' }}>
      
      {/* Header section */}
      <div>
        <h1 style={{ fontSize: '36px', fontWeight: '800', marginBottom: '8px' }}>
          Model <span className="gradient-text">Evaluation Metrics</span>
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '16px' }}>
          Explore classification stats and understand why each model achieved its specific level of accuracy.
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: '12px' }}>
        {[
          { id: 'summary', label: 'Overview & Explanations', icon: FileText },
          { id: 'confusion', label: 'Confusion Matrices', icon: CheckCircle },
          { id: 'roc', label: 'ROC AUC Curves', icon: TrendingUp },
          { id: 'distribution', label: 'Class Distributions', icon: BarChart2 }
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '10px 20px',
                borderRadius: '100px',
                border: 'none',
                background: isActive ? 'rgba(255, 148, 8, 0.12)' : 'transparent',
                color: isActive ? 'var(--color-orange)' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontWeight: '600',
                fontSize: '14px',
                transition: 'var(--transition-fast)',
                border: isActive ? '1px solid rgba(255, 148, 8, 0.3)' : '1px solid transparent'
              }}
              onMouseEnter={(e) => {
                if (!isActive) e.currentTarget.style.color = 'var(--text-primary)';
              }}
              onMouseLeave={(e) => {
                if (!isActive) e.currentTarget.style.color = 'var(--text-secondary)';
              }}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      {activeTab === 'summary' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
          
          {/* Summary Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px' }}>
            {metrics.map((row) => {
              const acc = parseFloat(row['Test Accuracy']);
              return (
                <div key={row.Model} className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px', borderLeftWidth: '4px', borderLeftColor: getMetricColor(row['Test Accuracy']) }}>
                  <span style={{ fontSize: '11px', fontWeight: '800', letterSpacing: '1px', textTransform: 'uppercase', color: 'var(--text-muted)' }}>{row.Model}</span>
                  <div>
                    <div style={{ fontSize: '28px', fontWeight: '800', color: getMetricColor(row['Test Accuracy']) }}>
                      {isNaN(acc) ? row['Test Accuracy'] : `${(acc * 100).toFixed(1)}%`}
                    </div>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Accuracy Score</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '8px', color: 'var(--text-muted)' }}>
                    <span>F1 (macro): {row['F1 Score (macro)'] ? `${(parseFloat(row['F1 Score (macro)']) * 100).toFixed(0)}%` : 'N/A'}</span>
                    <span>Classes: {row['No. Classes']}</span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Detailed explanations section (Why we got them) */}
          <div>
            <h2 style={{ fontSize: '20px', fontWeight: '700', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <HelpCircle size={20} color="var(--color-orange)" />
              Why We Got These Results
            </h2>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {metrics.map((row) => {
                const expKey = getExplanationKey(row.Model);
                const explanation = explanations ? explanations[expKey] : null;
                
                if (!explanation) return null;

                return (
                  <div key={row.Model} className="glass-panel" style={{ padding: '24px', display: 'grid', gridTemplateColumns: '240px 1fr', gap: '32px', alignItems: 'start' }}>
                    <div>
                      <h3 style={{ fontSize: '16px', fontWeight: '800', marginBottom: '8px' }}>{row.Model}</h3>
                      <div style={{ display: 'flex', gap: '16px', fontSize: '13px' }}>
                        <div>
                          <div style={{ fontSize: '18px', fontWeight: '700', color: getMetricColor(row['Test Accuracy']) }}>
                            {explanation && !isNaN(parseFloat(row['Test Accuracy'])) ? `${(parseFloat(row['Test Accuracy']) * 100).toFixed(1)}%` : row['Test Accuracy']}
                          </div>
                          <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>Accuracy</span>
                        </div>
                        <div style={{ borderRight: '1px solid rgba(255,255,255,0.08)' }} />
                        <div>
                          <div style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
                            {row['5-Fold CV Mean'] ? `${(parseFloat(row['5-Fold CV Mean']) * 100).toFixed(1)}%` : 'N/A'}
                          </div>
                          <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>CV Mean</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <div style={{ fontSize: '14px', fontWeight: '600', color: 'var(--color-orange)', marginBottom: '8px' }}>
                        {explanation.summary}
                      </div>
                      <p style={{ fontSize: '14px', lineHeight: '1.6', color: 'var(--text-secondary)' }}>
                        {explanation.why}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

        </div>
      )}

      {activeTab === 'confusion' && (
        <div className="glass-panel animate-fade-in" style={{ padding: '28px', textAlign: 'center' }}>
          <h2 style={{ fontSize: '20px', fontWeight: '700', marginBottom: '16px', textAlign: 'left' }}>Confusion Matrices</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '24px', textAlign: 'left' }}>
            Confusion matrices display true labels vs predicted labels, highlighting misclassifications across basic emotions, depression, and eating behaviors.
          </p>
          <img
            src="http://127.0.0.1:8000/static/confusion_matrices_all.png"
            alt="Confusion Matrices"
            style={{ maxWidth: '100%', height: 'auto', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.06)' }}
          />
        </div>
      )}

      {activeTab === 'roc' && (
        <div className="glass-panel animate-fade-in" style={{ padding: '28px', textAlign: 'center' }}>
          <h2 style={{ fontSize: '20px', fontWeight: '700', marginBottom: '16px', textAlign: 'left' }}>ROC AUC Curves</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '24px', textAlign: 'left' }}>
            Receiver Operating Characteristic (ROC) curves track true positive rates against false positive rates, showcasing the discriminative capability of the models.
          </p>
          <img
            src="http://127.0.0.1:8000/static/roc_auc_combined.png"
            alt="ROC AUC Curves"
            style={{ maxWidth: '100%', height: 'auto', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.06)' }}
          />
        </div>
      )}

      {activeTab === 'distribution' && (
        <div className="glass-panel animate-fade-in" style={{ padding: '28px', textAlign: 'center' }}>
          <h2 style={{ fontSize: '20px', fontWeight: '700', marginBottom: '16px', textAlign: 'left' }}>Class Distributions</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '24px', textAlign: 'left' }}>
            Understanding original dataset class imbalances explains potential biases or predictions trends.
          </p>
          <img
            src="http://127.0.0.1:8000/static/class_distribution.png"
            alt="Class Distribution"
            style={{ maxWidth: '100%', height: 'auto', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.06)' }}
          />
        </div>
      )}

    </div>
  );
}
