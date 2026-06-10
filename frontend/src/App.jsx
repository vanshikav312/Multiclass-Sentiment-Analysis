import React, { useState } from 'react';
import Analyzer from './components/Analyzer';
import Dashboard from './components/Dashboard';
import { Sparkles, BarChart2, Brain, Heart } from 'lucide-react';

export default function App() {
  const [currentPage, setCurrentPage] = useState('analyze'); // 'analyze' or 'metrics'

  return (
    <div style={{ display: 'flex', width: '100%', minHeight: '100vh', color: 'var(--text-primary)' }}>
      
      {/* Sidebar navigation */}
      <aside style={{
        width: '280px',
        background: 'rgba(16, 12, 8, 0.75)',
        borderRight: '1px solid rgba(202, 63, 22, 0.15)',
        backdropFilter: 'blur(20px)',
        padding: '32px 24px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        position: 'fixed',
        height: '100vh',
        zIndex: 10
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '40px' }}>
          
          {/* Logo / Title */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '12px',
              background: 'var(--gradient-3)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 15px rgba(149, 18, 44, 0.3)'
            }}>
              <Brain size={22} color="#fff" />
            </div>
            <div>
              <h2 style={{ fontSize: '20px', fontWeight: '800', letterSpacing: '-0.02em' }}>
                <span className="gradient-text">Mind</span>Scan
              </h2>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Sentiment Intelligence
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <button
              onClick={() => setCurrentPage('analyze')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '14px 20px',
                borderRadius: '12px',
                border: 'none',
                background: currentPage === 'analyze' ? 'var(--gradient-3)' : 'transparent',
                color: currentPage === 'analyze' ? '#fff' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontWeight: '600',
                fontFamily: 'var(--font-display)',
                fontSize: '15px',
                textAlign: 'left',
                boxShadow: currentPage === 'analyze' ? '0 4px 20px rgba(149, 18, 44, 0.25)' : 'none',
                transition: 'var(--transition-fast)'
              }}
              onMouseEnter={(e) => {
                if (currentPage !== 'analyze') {
                  e.currentTarget.style.background = 'rgba(255, 148, 8, 0.08)';
                  e.currentTarget.style.color = 'var(--text-primary)';
                }
              }}
              onMouseLeave={(e) => {
                if (currentPage !== 'analyze') {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.color = 'var(--text-secondary)';
                }
              }}
            >
              <Sparkles size={18} />
              Analyze Sentence
            </button>

            <button
              onClick={() => setCurrentPage('metrics')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '14px 20px',
                borderRadius: '12px',
                border: 'none',
                background: currentPage === 'metrics' ? 'var(--gradient-3)' : 'transparent',
                color: currentPage === 'metrics' ? '#fff' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontWeight: '600',
                fontFamily: 'var(--font-display)',
                fontSize: '15px',
                textAlign: 'left',
                boxShadow: currentPage === 'metrics' ? '0 4px 20px rgba(149, 18, 44, 0.25)' : 'none',
                transition: 'var(--transition-fast)'
              }}
              onMouseEnter={(e) => {
                if (currentPage !== 'metrics') {
                  e.currentTarget.style.background = 'rgba(255, 148, 8, 0.08)';
                  e.currentTarget.style.color = 'var(--text-primary)';
                }
              }}
              onMouseLeave={(e) => {
                if (currentPage !== 'metrics') {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.color = 'var(--text-secondary)';
                }
              }}
            >
              <BarChart2 size={18} />
              Evaluation Metrics
            </button>
          </nav>

        </div>

        {/* Footer Info */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '11px', color: 'var(--text-muted)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span>Made with</span>
            <Heart size={10} color="var(--color-crimson)" style={{ fill: 'var(--color-crimson)' }} />
            <span>for ML Analysis</span>
          </div>
          <div>v1.0.0 | React + FastAPI</div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main style={{
        flex: 1,
        marginLeft: '280px',
        padding: '48px 64px',
        minHeight: '100vh',
        overflowY: 'auto'
      }}>
        {currentPage === 'analyze' ? <Analyzer /> : <Dashboard />}
      </main>

    </div>
  );
}
