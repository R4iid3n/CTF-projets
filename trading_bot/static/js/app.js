// Trading Bots Dashboard - Client-side JavaScript

// WebSocket connection
const socket = io();

// Chart instance
let equityChart = null;
let currentCodeBotId = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Dashboard initialized');
    initializeWebSocket();
    initializeChart();
    loadInitialData();
    
    // Refresh every 30 seconds as backup
    setInterval(refreshData, 30000);
});

// WebSocket handlers
function initializeWebSocket() {
    socket.on('connect', function() {
        console.log('✅ Connected to server');
        socket.emit('request_update');
    });
    
    socket.on('connected', function(data) {
        console.log('Server message:', data.message);
    });
    
    socket.on('bot_update', function(data) {
        updateBotDisplay(data.bot_id, data.data);
    });
    
    socket.on('full_update', function(data) {
        updateAllBots(data.bots);
        updateGlobalStats(data.comparison);
    });
    
    socket.on('disconnect', function() {
        console.log('❌ Disconnected from server');
    });
}

// Load initial data
function loadInitialData() {
    fetch('/api/bots')
        .then(response => response.json())
        .then(data => {
            updateAllBots(data.bots);
            updateGlobalStats(data.comparison);
        })
        .catch(error => console.error('Error loading data:', error));
}

// Refresh data
function refreshData() {
    socket.emit('request_update');
}

// Update all bots
function updateAllBots(bots) {
    for (const [botId, botData] of Object.entries(bots)) {
        updateBotDisplay(botId, botData);
    }
    updateEquityChart(bots);
}

// Update single bot display
function updateBotDisplay(botId, data) {
    // Status
    const statusEl = document.getElementById(`status-${botId}`);
    if (statusEl) {
        statusEl.textContent = data.running ? '🟢 Running' : '⚫ Stopped';
        statusEl.className = data.running ? 'bot-status status-running' : 'bot-status status-stopped';
    }
    
    // Philosophy (only on first load)
    const philosophyEl = document.getElementById(`philosophy-${botId}`);
    if (philosophyEl && philosophyEl.children.length === 0 && data.philosophy) {
        data.philosophy.forEach(point => {
            const li = document.createElement('li');
            li.textContent = point;
            philosophyEl.appendChild(li);
        });
    }
    
    // Metrics
    updateElement(`capital-${botId}`, `$${data.capital.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`);
    
    const winnings = data.cumulative_winnings;
    const winningsColor = winnings >= 0 ? 'var(--accent-success)' : 'var(--accent-danger)';
    updateElement(`winnings-${botId}`, `$${winnings.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`, winningsColor);
    
    updateElement(`trades-${botId}`, data.trades_executed);
    updateElement(`winrate-${botId}`, `${data.win_rate}%`);
    
    // Logs
    const logEl = document.getElementById(`log-${botId}`);
    if (logEl && data.log_messages && data.log_messages.length > 0) {
        logEl.innerHTML = data.log_messages.slice(-10).map(msg => `<div>${msg}</div>`).join('');
    }
}

// Update global stats
function updateGlobalStats(stats) {
    updateElement('total-capital', `$${stats.total_capital.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`);
    
    const totalWinnings = stats.total_winnings;
    const winningsColor = totalWinnings >= 0 ? 'var(--accent-success)' : 'var(--accent-danger)';
    updateElement('total-winnings', `$${totalWinnings.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`, winningsColor);
    
    updateElement('total-trades', stats.total_trades);
    
    if (stats.best_performer) {
        updateElement('best-performer', `${stats.best_performer.name} ($${stats.best_performer.winnings.toFixed(2)})`);
    }
}

// Helper to update element
function updateElement(id, text, color = null) {
    const el = document.getElementById(id);
    if (el) {
        el.textContent = text;
        if (color) {
            el.style.color = color;
        }
    }
}

// Initialize equity chart
function initializeChart() {
    const ctx = document.getElementById('equityChart');
    if (!ctx) return;
    
    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'v13 FIXED',
                    data: [],
                    borderColor: '#ff4757',
                    backgroundColor: 'rgba(255, 71, 87, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'v13.5 ENHANCED',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'v14 TRUE METHOD',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#b0b8d4',
                        font: {
                            size: 14
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(26, 35, 66, 0.95)',
                    titleColor: '#00d4ff',
                    bodyColor: '#b0b8d4',
                    borderColor: '#2a3558',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute'
                    },
                    grid: {
                        color: '#2a3558'
                    },
                    ticks: {
                        color: '#6b7399'
                    }
                },
                y: {
                    beginAtZero: false,
                    grid: {
                        color: '#2a3558'
                    },
                    ticks: {
                        color: '#6b7399',
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// Update equity chart
function updateEquityChart(bots) {
    if (!equityChart) return;
    
    const botMapping = {
        'v13': 0,
        'v13_5': 1,
        'v14': 2
    };
    
    for (const [botId, botData] of Object.entries(bots)) {
        const datasetIndex = botMapping[botId];
        if (datasetIndex === undefined) continue;
        
        if (botData.equity_curve && botData.equity_curve.length > 0) {
            equityChart.data.datasets[datasetIndex].data = botData.equity_curve.map(point => ({
                x: new Date(point.timestamp),
                y: point.value
            }));
        }
    }
    
    equityChart.update('none'); // Update without animation for smooth real-time
}

// Bot control functions
async function startBot(botId) {
    try {
        const response = await fetch(`/api/bot/${botId}/start`, {
            method: 'POST'
        });
        const data = await response.json();
        console.log(`Started ${botId}:`, data);
        setTimeout(refreshData, 500);
    } catch (error) {
        console.error(`Error starting ${botId}:`, error);
    }
}

async function stopBot(botId) {
    try {
        const response = await fetch(`/api/bot/${botId}/stop`, {
            method: 'POST'
        });
        const data = await response.json();
        console.log(`Stopped ${botId}:`, data);
        setTimeout(refreshData, 500);
    } catch (error) {
        console.error(`Error stopping ${botId}:`, error);
    }
}

async function startAll() {
    try {
        const response = await fetch('/api/bots/start_all', {
            method: 'POST'
        });
        const data = await response.json();
        console.log('Started all bots:', data);
        setTimeout(refreshData, 500);
    } catch (error) {
        console.error('Error starting all bots:', error);
    }
}

async function stopAll() {
    try {
        const response = await fetch('/api/bots/stop_all', {
            method: 'POST'
        });
        const data = await response.json();
        console.log('Stopped all bots:', data);
        setTimeout(refreshData, 500);
    } catch (error) {
        console.error('Error stopping all bots:', error);
    }
}

// Code viewer functions
async function viewCode(botId) {
    currentCodeBotId = botId;
    const modal = document.getElementById('codeModal');
    const title = document.getElementById('codeModalTitle');
    const content = document.getElementById('codeContent');
    
    // Show modal
    modal.style.display = 'block';
    
    // Update title
    const botNames = {
        'v13': 'v13 FIXED',
        'v13_5': 'v13.5 ENHANCED',
        'v14': 'v14 TRUE METHOD'
    };
    title.textContent = `${botNames[botId]} - Source Code`;
    
    // Load code
    content.textContent = 'Loading code...';
    
    try {
        const response = await fetch(`/api/bot/${botId}/code`);
        const data = await response.json();
        
        if (data.code) {
            content.textContent = data.code;
        } else {
            content.textContent = `Error: ${data.error || 'Unable to load code'}`;
        }
    } catch (error) {
        content.textContent = `Error loading code: ${error.message}`;
        console.error('Error fetching code:', error);
    }
}

function closeCodeModal() {
    const modal = document.getElementById('codeModal');
    modal.style.display = 'none';
}

function downloadCode() {
    if (!currentCodeBotId) return;
    
    const content = document.getElementById('codeContent').textContent;
    const filenames = {
        'v13': 'yuichi_bot_v13_fixed.py',
        'v13_5': 'yuichi_bot_v13_5_enhanced.py',
        'v14': 'yuichi_bot_v14_true_method.py'
    };
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filenames[currentCodeBotId] || 'bot_code.py';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('codeModal');
    if (event.target === modal) {
        closeCodeModal();
    }
}

// Log connection status
console.log('🎮 Trading Bots Dashboard loaded');
