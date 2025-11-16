// VPS Trading Bots Dashboard - Real-time monitoring

let positionsChart = null;
let currentLogBot = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Dashboard initialized');
    initializeChart();
    updateData();
    
    // Auto-refresh every 3 seconds
    setInterval(updateData, 3000);
});

// Update all data
async function updateData() {
    try {
        const response = await fetch('/api/bots');
        const data = await response.json();
        
        updateGlobalStats(data.summary);
        updateBotCards(data.bots);
        updatePositionsChart(data.bots);
        updateVPSStatus(data.summary.vps_connected);
        
    } catch (error) {
        console.error('Update error:', error);
    }
}

// Update VPS status indicator
function updateVPSStatus(connected) {
    const statusEl = document.getElementById('vps-status');
    if (connected) {
        statusEl.textContent = '🟢 Connected';
        statusEl.style.background = 'rgba(0, 255, 136, 0.2)';
        statusEl.style.color = 'var(--accent-success)';
    } else {
        statusEl.textContent = '🔴 Disconnected';
        statusEl.style.background = 'rgba(255, 71, 87, 0.2)';
        statusEl.style.color = 'var(--accent-danger)';
    }
}

// Update global stats
function updateGlobalStats(summary) {
    const totalWinnings = summary.total_winnings || 0;
    const winningsColor = totalWinnings >= 0 ? 'var(--accent-success)' : 'var(--accent-danger)';
    
    updateElement('total-capital', `$${(summary.total_capital || 0).toFixed(2)}`);
    updateElement('total-winnings', `$${totalWinnings.toFixed(2)}`, winningsColor);
    updateElement('total-trades', summary.total_trades || 0);
    
    if (summary.best_performer) {
        updateElement('best-performer', 
            `${summary.best_performer.name} ($${summary.best_performer.winnings.toFixed(2)})`
        );
    }
}

// Update bot cards
function updateBotCards(bots) {
    const container = document.getElementById('bots-container');
    
    // Clear and rebuild
    container.innerHTML = '';
    
    bots.forEach(bot => {
        const card = createBotCard(bot);
        container.appendChild(card);
    });
}

// Create bot card element
function createBotCard(bot) {
    const card = document.createElement('div');
    card.className = 'bot-card' + (bot.position_active ? ' active' : '');
    
    const winnings = bot.cumulative_winnings || 0;
    const winningsClass = winnings >= 0 ? 'positive' : 'negative';
    
    card.innerHTML = `
        <div class="bot-header">
            <div class="bot-name" style="color: ${bot.color}">${bot.display_name || bot.bot_name}</div>
            <div class="bot-status ${bot.running ? 'status-running' : 'status-stopped'}">
                ${bot.running ? '🟢 Running' : '⚫ Stopped'}
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-label">Capital</div>
                <div class="metric-value">$${(bot.capital || 0).toFixed(2)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Winnings</div>
                <div class="metric-value ${winningsClass}">$${winnings.toFixed(2)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Trades Today</div>
                <div class="metric-value">${bot.trades_executed_today || 0}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Battle Step</div>
                <div class="metric-value">${bot.battle_step || 0} / ${bot.martingale_step || 0}</div>
            </div>
        </div>
        
        ${bot.position_active ? createPositionDisplay(bot) : '<div style="text-align:center; padding:20px; color:var(--text-muted)">No active position</div>'}
        
        <button class="btn-logs" onclick="viewLogs('${bot.bot_id}', '${bot.display_name}')">
            📋 View Logs
        </button>
    `;
    
    return card;
}

// Create position display
function createPositionDisplay(bot) {
    const posType = bot.side || 'N/A';
    const posClass = posType.toLowerCase() === 'buy' ? 'long' : 'short';
    
    return `
        <div class="position-display">
            <div class="position-header">
                <div class="position-type ${posClass}">
                    ${posType.toUpperCase()} POSITION
                </div>
                <div style="color: var(--text-secondary)">
                    ${bot.position_size_main ? `$${bot.position_size_main.toFixed(2)}` : 'N/A'}
                    ${bot.position_size_hedge ? ` + $${bot.position_size_hedge.toFixed(2)} hedge` : ''}
                </div>
            </div>
            
            <div class="position-prices">
                <div class="price-item">
                    <div class="price-label">Entry</div>
                    <div class="price-value">${bot.entry_price ? `$${bot.entry_price.toFixed(2)}` : 'N/A'}</div>
                </div>
                <div class="price-item">
                    <div class="price-label">Stop Loss</div>
                    <div class="price-value" style="color: var(--accent-danger)">
                        ${bot.stop_loss ? `$${bot.stop_loss.toFixed(2)}` : 'N/A'}
                    </div>
                </div>
                <div class="price-item">
                    <div class="price-label">Take Profit</div>
                    <div class="price-value" style="color: var(--accent-success)">
                        ${bot.take_profit ? `$${bot.take_profit.toFixed(2)}` : 'N/A'}
                    </div>
                </div>
            </div>
            
            ${bot.game_state ? `
                <div class="game-state ${bot.game_state.toLowerCase().replace(' ', '_')}">
                    ${bot.game_state}
                    ${bot.trap_type && bot.trap_type !== 'none' ? ` • ${bot.trap_type}` : ''}
                </div>
            ` : ''}
            
            ${bot.battle_pnl !== undefined ? `
                <div style="margin-top: 10px; text-align: center; font-weight: bold; font-size: 1.1rem; color: ${bot.battle_pnl >= 0 ? 'var(--accent-success)' : 'var(--accent-danger)'}">
                    Battle P&L: $${bot.battle_pnl.toFixed(2)}
                </div>
            ` : ''}
        </div>
    `;
}

// Initialize positions chart
function initializeChart() {
    const ctx = document.getElementById('positionsChart');
    if (!ctx) return;
    
    positionsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Current Position Size',
                data: [],
                backgroundColor: [],
                borderColor: [],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 35, 66, 0.95)',
                    titleColor: '#00d4ff',
                    bodyColor: '#b0b8d4',
                    borderColor: '#2a3558',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return `Position: $${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: '#2a3558' },
                    ticks: { color: '#6b7399' }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: '#2a3558' },
                    ticks: {
                        color: '#6b7399',
                        callback: function(value) {
                            return '$' + value.toFixed(0);
                        }
                    }
                }
            }
        }
    });
}

// Update positions chart
function updatePositionsChart(bots) {
    if (!positionsChart) return;
    
    const labels = [];
    const data = [];
    const bgColors = [];
    const borderColors = [];
    
    bots.forEach(bot => {
        if (bot.position_active && bot.position_size_main) {
            labels.push(bot.display_name || bot.bot_name);
            data.push(bot.position_size_main + (bot.position_size_hedge || 0));
            
            const color = bot.color || '#00d4ff';
            bgColors.push(color + '40');
            borderColors.push(color);
        }
    });
    
    positionsChart.data.labels = labels;
    positionsChart.data.datasets[0].data = data;
    positionsChart.data.datasets[0].backgroundColor = bgColors;
    positionsChart.data.datasets[0].borderColor = borderColors;
    
    positionsChart.update('none');
}

// View logs
async function viewLogs(botId, botName) {
    currentLogBot = botId;
    
    const modal = document.getElementById('logModal');
    const title = document.getElementById('logModalTitle');
    const content = document.getElementById('logContent');
    
    modal.style.display = 'block';
    title.textContent = `${botName} - Logs`;
    content.textContent = 'Loading logs...';
    
    try {
        const response = await fetch(`/api/bot/${botId}`);
        const bot = await response.json();
        
        if (bot.recent_logs) {
            content.textContent = bot.recent_logs;
        } else {
            content.textContent = 'No logs available';
        }
    } catch (error) {
        content.textContent = `Error loading logs: ${error.message}`;
    }
}

// Close log modal
function closeLogModal() {
    document.getElementById('logModal').style.display = 'none';
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

// Close modal on outside click
window.onclick = function(event) {
    const modal = document.getElementById('logModal');
    if (event.target === modal) {
        closeLogModal();
    }
}

console.log('🎮 VPS Dashboard loaded');
