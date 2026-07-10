import React, { useState, useEffect, useCallback } from 'react';
import { 
  Wind, Sun, Zap, Activity, TrendingUp, AlertTriangle, 
  BarChart3, Calendar, Brain, ChevronRight, Menu,
  Battery, Gauge, Target, BarChart2
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, AreaChart, Area, Bar, ComposedChart,
  ReferenceLine
} from 'recharts';
import axios from 'axios';

const API_URL = process.env.REACT_APP_BACKEND_URL || '';

// Custom Tooltip Component
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="custom-tooltip bg-zinc-900/95 border border-white/10 rounded-lg p-3">
        <p className="text-zinc-400 text-xs mb-2">{label}</p>
        {payload.map((entry, index) => (
          <p key={index} className="data-value text-sm" style={{ color: entry.color }}>
            {entry.name}: <span className="font-semibold">{typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}</span>
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// Metric Card Component for Model Performance
const MetricCard = ({ label, value, unit, subtext, color, icon: Icon }) => (
  <div className="card p-4 animate-fade-in">
    <div className="flex items-center gap-2 mb-2">
      {Icon && <Icon className={`w-4 h-4 text-${color}`} />}
      <span className="text-zinc-500 text-xs uppercase tracking-wide">{label}</span>
    </div>
    <p className={`text-2xl font-heading font-bold text-${color} data-value`}>
      {value}<span className="text-sm text-zinc-500 ml-1">{unit}</span>
    </p>
    {subtext && <p className="text-zinc-500 text-xs mt-1">{subtext}</p>}
  </div>
);

// Stat Card Component
const StatCard = ({ icon: Icon, label, value, unit, color, trend }) => (
  <div className="card p-5 animate-fade-in" data-testid={`stat-card-${label.toLowerCase().replace(/\s/g, '-')}`}>
    <div className="flex items-start justify-between mb-3">
      <div className={`p-2 rounded-lg bg-${color}/10`}>
        <Icon className={`w-5 h-5 text-${color}`} />
      </div>
      {trend && (
        <span className={`text-xs px-2 py-1 rounded-full ${trend > 0 ? 'bg-success/10 text-success' : 'bg-error/10 text-error'}`}>
          {trend > 0 ? '+' : ''}{trend}%
        </span>
      )}
    </div>
    <p className="text-zinc-500 text-sm font-body mb-1">{label}</p>
    <p className="text-2xl font-heading font-bold text-white leading-none data-value">
      {value}<span className="text-lg text-zinc-500 ml-1">{unit}</span>
    </p>
  </div>
);

// Live Indicator
const LiveIndicator = ({ data }) => (
  <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-success/10 border border-success/20">
    <div className="w-2 h-2 rounded-full bg-success pulse-live" />
    <span className="text-success text-xs font-mono">LIVE</span>
    <span className="text-white text-xs font-mono">{data?.total_load?.toFixed(1) || '---'} MW</span>
  </div>
);

// Navigation Items
const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
  { id: 'wind', label: 'Wind Power', icon: Wind },
  { id: 'solar', label: 'Solar Power', icon: Sun },
  { id: 'machines', label: 'Machine Consumption', icon: Zap },
  { id: 'performance', label: 'Model Performance', icon: Target },
];

function App() {
  const [activeView, setActiveView] = useState('dashboard');
  const [selectedDate, setSelectedDate] = useState('02-01-2022');
  const [availableDates, setAvailableDates] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  
  // Data states
  const [dashboardData, setDashboardData] = useState(null);
  const [windData, setWindData] = useState(null);
  const [solarData, setSolarData] = useState(null);
  const [machineData, setMachineData] = useState(null);
  const [performanceData, setPerformanceData] = useState(null);
  const [aiInsight, setAiInsight] = useState('');
  const [liveData, setLiveData] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Model Training states
  const [modelStatus, setModelStatus] = useState({ wind_trained: false, solar_trained: false });
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [trainedPredictions, setTrainedPredictions] = useState(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('gradient_boosting');
  const [featureImportance, setFeatureImportance] = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [comparing, setComparing] = useState(false);

  // Fetch available dates and model status
  useEffect(() => {
    axios.get(`${API_URL}/api/dates`).then(res => {
      setAvailableDates(res.data.dates || []);
    }).catch(console.error);
    
    // Check if models are already trained
    axios.get(`${API_URL}/api/model-status`).then(res => {
      setModelStatus(res.data);
    }).catch(console.error);
  }, []);

  // Fetch data based on selected date and view
  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      // Fetch energy data based on active view
      if (activeView === 'dashboard') {
        const [windRes, solarRes, dashRes, machRes] = await Promise.all([
          axios.post(`${API_URL}/api/wind-prediction`, { date: selectedDate }),
          axios.post(`${API_URL}/api/solar-prediction`, { date: selectedDate }),
          axios.post(`${API_URL}/api/dashboard-summary`, { date: selectedDate }),
          axios.post(`${API_URL}/api/machine-consumption`, { date: '01-01-2023' })
        ]);
        setWindData(windRes.data);
        setSolarData(solarRes.data);
        setDashboardData(dashRes.data);
        setMachineData(machRes.data);
      } else if (activeView === 'wind') {
        const res = await axios.post(`${API_URL}/api/wind-prediction`, { date: selectedDate });
        setWindData(res.data);
      } else if (activeView === 'solar') {
        const res = await axios.post(`${API_URL}/api/solar-prediction`, { date: selectedDate });
        setSolarData(res.data);
      } else if (activeView === 'machines') {
        const res = await axios.post(`${API_URL}/api/machine-consumption`, { date: '01-01-2023' });
        setMachineData(res.data);
      } else if (activeView === 'performance') {
        const res = await axios.post(`${API_URL}/api/model-performance`, { date: selectedDate });
        setPerformanceData(res.data);
        // Fetch feature importance if models are trained
        axios.get(`${API_URL}/api/feature-importance`)
          .then(fiRes => { if (!fiRes.data.error) setFeatureImportance(fiRes.data); })
          .catch(() => {});
        // Fetch experiment history
        axios.get(`${API_URL}/api/experiments`)
          .then(expRes => setExperiments(expRes.data.experiments || []))
          .catch(() => {});
      }
      
      setLoading(false);
      
      // Fetch AI insight asynchronously (non-blocking)
      axios.post(`${API_URL}/api/ai-insights`, { 
        date: selectedDate, 
        context: activeView 
      }).then(aiRes => setAiInsight(aiRes.data.insight))
        .catch(() => setAiInsight('Analyzing energy patterns...'));
      
    } catch (err) {
      console.error('Fetch error:', err);
      setLoading(false);
    }
  }, [selectedDate, activeView]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Train model function
  const trainModels = async (modelType = 'both') => {
    setTrainingInProgress(true);
    try {
      const res = await axios.post(`${API_URL}/api/train-model`, {
        model_type: modelType,
        algorithm: selectedAlgorithm,
        test_size: 0.2
      });
      
      // Update model status
      const statusRes = await axios.get(`${API_URL}/api/model-status`);
      setModelStatus(statusRes.data);
      
      // Fetch improved predictions for current date
      const predRes = await axios.post(`${API_URL}/api/predict-with-trained`, { date: selectedDate });
      setTrainedPredictions(predRes.data);
      
      // Refresh performance data
      const perfRes = await axios.post(`${API_URL}/api/model-performance`, { date: selectedDate });
      setPerformanceData(perfRes.data);
      
      // Fetch feature importance from newly trained models
      const fiRes = await axios.get(`${API_URL}/api/feature-importance`);
      if (!fiRes.data.error) setFeatureImportance(fiRes.data);
      
      // Refresh experiment history
      const expRes = await axios.get(`${API_URL}/api/experiments`);
      setExperiments(expRes.data.experiments || []);
      
      return res.data;
    } catch (err) {
      console.error('Training error:', err);
    } finally {
      setTrainingInProgress(false);
    }
  };

  // Fetch trained predictions when date changes
  useEffect(() => {
    if (modelStatus.wind_trained || modelStatus.solar_trained) {
      axios.post(`${API_URL}/api/predict-with-trained`, { date: selectedDate })
        .then(res => setTrainedPredictions(res.data))
        .catch(console.error);
    }
  }, [selectedDate, modelStatus.wind_trained, modelStatus.solar_trained]);

  // WebSocket for real-time data
  useEffect(() => {
    const wsUrl = `${API_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/api/ws/realtime`;
    let ws;
    
    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setLiveData(data);
      };
      ws.onerror = () => {
        // Fallback to simulated data
        const interval = setInterval(() => {
          setLiveData({
            wind_power: Math.random() * 60 + 20,
            solar_power: Math.random() * 40 + 30,
            grid_frequency: 49.95 + Math.random() * 0.1,
            total_load: Math.random() * 100 + 100
          });
        }, 2000);
        return () => clearInterval(interval);
      };
    } catch {
      // Fallback simulation
      const interval = setInterval(() => {
        setLiveData({
          wind_power: Math.random() * 60 + 20,
          solar_power: Math.random() * 40 + 30,
          grid_frequency: 49.95 + Math.random() * 0.1,
          total_load: Math.random() * 100 + 100
        });
      }, 2000);
      return () => clearInterval(interval);
    }
    
    return () => ws?.close();
  }, []);

  const pieColors = ['#0EA5E9', '#F59E0B'];
  const pieData = dashboardData ? [
    { name: 'Wind', value: dashboardData.wind_total },
    { name: 'Solar', value: dashboardData.solar_total }
  ] : [];

  return (
    <div className="flex h-screen bg-background" data-testid="app-container">
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? 'w-64' : 'w-20'} bg-surface border-r border-white/10 flex flex-col transition-all duration-300`}>
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-wind to-solar flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            {sidebarOpen && (
              <div>
                <h1 className="font-heading font-bold text-xl text-white">GridWise</h1>
                <p className="text-xs text-zinc-500">Energy Intelligence</p>
              </div>
            )}
          </div>
        </div>
        
        <nav className="flex-1 p-4">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveView(item.id)}
              data-testid={`nav-${item.id}`}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg mb-2 transition-all
                ${activeView === item.id 
                  ? 'bg-primary/10 text-primary border border-primary/20' 
                  : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
            >
              <item.icon className="w-5 h-5" />
              {sidebarOpen && <span className="font-medium">{item.label}</span>}
              {sidebarOpen && activeView === item.id && <ChevronRight className="w-4 h-4 ml-auto" />}
            </button>
          ))}
        </nav>
        
        <div className="p-4 border-t border-white/10">
          <button 
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="w-full flex items-center justify-center p-2 rounded-lg hover:bg-white/5 text-zinc-400"
          >
            <Menu className="w-5 h-5" />
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        {/* Header */}
        <header className="sticky top-0 z-10 glass border-b border-white/5 px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-heading font-bold text-2xl text-white">
                {navItems.find(n => n.id === activeView)?.label || 'Dashboard'}
              </h2>
              <p className="text-zinc-500 text-sm">Real-time energy monitoring & analytics</p>
            </div>
            <div className="flex items-center gap-4">
              <LiveIndicator data={liveData} />
              <div className="flex items-center gap-2">
                <Calendar className="w-4 h-4 text-zinc-400" />
                <select
                  value={selectedDate}
                  onChange={(e) => setSelectedDate(e.target.value)}
                  data-testid="date-selector"
                  className="bg-surface border border-white/10 rounded-lg px-3 py-2 text-sm text-white font-mono focus:outline-none focus:border-primary"
                >
                  {availableDates.map(date => (
                    <option key={date} value={date}>{date}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="p-8">
          {loading ? (
            <div className="flex items-center justify-center h-64">
              <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            </div>
          ) : (
            <>
              {/* AI Insights Panel */}
              <div className="card p-5 mb-6 border-l-4 border-l-primary" data-testid="ai-insights-panel">
                <div className="flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-primary/10">
                    <Brain className="w-5 h-5 text-primary" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-heading font-semibold text-white mb-1">AI Insights</h3>
                    <p className="text-zinc-400 text-sm leading-relaxed">{aiInsight || 'Analyzing energy data...'}</p>
                  </div>
                </div>
              </div>

              {/* Dashboard View */}
              {activeView === 'dashboard' && (
                <div className="space-y-6" data-testid="dashboard-view">
                  {/* Live Stats */}
                  <div className="grid grid-cols-4 gap-4">
                    <StatCard 
                      icon={Wind} 
                      label="Live Wind Power" 
                      value={liveData?.wind_power?.toFixed(1) || '--'} 
                      unit="kW" 
                      color="wind" 
                    />
                    <StatCard 
                      icon={Sun} 
                      label="Live Solar Power" 
                      value={liveData?.solar_power?.toFixed(1) || '--'} 
                      unit="kW" 
                      color="solar" 
                    />
                    <StatCard 
                      icon={Gauge} 
                      label="Grid Frequency" 
                      value={liveData?.grid_frequency?.toFixed(3) || '--'} 
                      unit="Hz" 
                      color="success" 
                    />
                    <StatCard 
                      icon={Battery} 
                      label="Total Generation" 
                      value={dashboardData?.combined_total?.toFixed(0) || '--'} 
                      unit="kW" 
                      color="primary" 
                    />
                  </div>

                  {/* Charts Row */}
                  <div className="grid grid-cols-3 gap-6">
                    {/* Combined Generation Chart */}
                    <div className="card p-5 col-span-2" data-testid="combined-generation-chart">
                      <h3 className="font-heading font-semibold text-white mb-4">Energy Generation Overview</h3>
                      <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={windData?.data || []}>
                          <defs>
                            <linearGradient id="windGradient" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#0EA5E9" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#0EA5E9" stopOpacity={0}/>
                            </linearGradient>
                            <linearGradient id="solarGradient" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#F59E0B" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <XAxis dataKey="time" stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                          <YAxis stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                          <Tooltip content={<CustomTooltip />} />
                          <Area type="monotone" dataKey="actual" name="Wind" stroke="#0EA5E9" fill="url(#windGradient)" strokeWidth={2} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Pie Chart */}
                    <div className="card p-5" data-testid="energy-distribution-chart">
                      <h3 className="font-heading font-semibold text-white mb-4">Energy Distribution</h3>
                      <ResponsiveContainer width="100%" height={200}>
                        <PieChart>
                          <Pie
                            data={pieData}
                            cx="50%"
                            cy="50%"
                            innerRadius={50}
                            outerRadius={80}
                            dataKey="value"
                            paddingAngle={2}
                          >
                            {pieData.map((_, index) => (
                              <Cell key={`cell-${index}`} fill={pieColors[index]} />
                            ))}
                          </Pie>
                          <Tooltip content={<CustomTooltip />} />
                        </PieChart>
                      </ResponsiveContainer>
                      <div className="flex justify-center gap-6 mt-4">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-wind" />
                          <span className="text-sm text-zinc-400">Wind ({dashboardData?.wind_percentage || 0}%)</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-solar" />
                          <span className="text-sm text-zinc-400">Solar ({dashboardData?.solar_percentage || 0}%)</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Anomalies Section */}
                  {machineData?.stats?.anomalies?.length > 0 && (
                    <div className="card p-5" data-testid="anomalies-panel">
                      <div className="flex items-center gap-3 mb-4">
                        <AlertTriangle className="w-5 h-5 text-warning" />
                        <h3 className="font-heading font-semibold text-white">Detected Anomalies</h3>
                        <span className="px-2 py-1 bg-warning/10 text-warning text-xs rounded-full">
                          {machineData.stats.anomaly_count} alerts
                        </span>
                      </div>
                      <div className="grid grid-cols-5 gap-3">
                        {machineData.stats.anomalies.map((anomaly, i) => (
                          <div key={i} className="bg-warning/5 border border-warning/20 rounded-lg p-3">
                            <p className="text-white font-medium text-sm">{anomaly.machine}</p>
                            <p className="text-warning text-xs mt-1">{anomaly.type}</p>
                            <p className="text-white font-mono text-lg mt-1">{anomaly.value}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Wind Power View */}
              {activeView === 'wind' && (
                <div className="space-y-6" data-testid="wind-view">
                  <div className="grid grid-cols-3 gap-4">
                    <StatCard icon={TrendingUp} label="Average Power" value={windData?.stats?.average || 0} unit="kW" color="wind" />
                    <StatCard icon={Activity} label="Peak Power" value={windData?.stats?.maximum || 0} unit="kW" color="wind" />
                    <StatCard icon={Zap} label="Total Generation" value={windData?.stats?.total || 0} unit="kWh" color="wind" />
                  </div>
                  
                  <div className="card p-5" data-testid="wind-prediction-chart">
                    <h3 className="font-heading font-semibold text-white mb-4">Wind Power: Actual vs Predicted</h3>
                    <ResponsiveContainer width="100%" height={400}>
                      <LineChart data={windData?.data || []}>
                        <XAxis dataKey="time" stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                        <YAxis stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                        <Tooltip content={<CustomTooltip />} />
                        <Line type="monotone" dataKey="actual" name="Actual" stroke="#0EA5E9" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="predicted" name="Predicted" stroke="#0EA5E9" strokeWidth={2} strokeDasharray="5 5" dot={false} opacity={0.6} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Solar Power View */}
              {activeView === 'solar' && (
                <div className="space-y-6" data-testid="solar-view">
                  <div className="grid grid-cols-3 gap-4">
                    <StatCard icon={TrendingUp} label="Average Power" value={solarData?.stats?.average || 0} unit="kW" color="solar" />
                    <StatCard icon={Activity} label="Peak Power" value={solarData?.stats?.maximum || 0} unit="kW" color="solar" />
                    <StatCard icon={Sun} label="Total Generation" value={solarData?.stats?.total || 0} unit="kWh" color="solar" />
                  </div>
                  
                  <div className="card p-5" data-testid="solar-prediction-chart">
                    <h3 className="font-heading font-semibold text-white mb-4">Solar Power: Actual vs Predicted</h3>
                    <ResponsiveContainer width="100%" height={400}>
                      <LineChart data={solarData?.data || []}>
                        <XAxis dataKey="time" stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                        <YAxis stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                        <Tooltip content={<CustomTooltip />} />
                        <Line type="monotone" dataKey="actual" name="Actual" stroke="#F59E0B" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="predicted" name="Predicted" stroke="#F59E0B" strokeWidth={2} strokeDasharray="5 5" dot={false} opacity={0.6} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Machine Consumption View */}
              {activeView === 'machines' && (
                <div className="space-y-6" data-testid="machines-view">
                  <div className="grid grid-cols-3 gap-4">
                    <StatCard icon={Zap} label="Total Consumption" value={machineData?.stats?.total_consumption?.toFixed(0) || 0} unit="kWh" color="consumption" />
                    <StatCard icon={AlertTriangle} label="Anomalies Detected" value={machineData?.stats?.anomaly_count || 0} unit="alerts" color="warning" />
                    <StatCard icon={Activity} label="Machines Monitored" value="5" unit="units" color="success" />
                  </div>

                  {/* Energy Chart */}
                  <div className="card p-5" data-testid="machine-energy-chart">
                    <h3 className="font-heading font-semibold text-white mb-4">Machine Energy Consumption</h3>
                    <ResponsiveContainer width="100%" height={350}>
                      <LineChart data={machineData?.data || []}>
                        <XAxis dataKey="time" stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                        <YAxis stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                        <Tooltip content={<CustomTooltip />} />
                        <Line type="monotone" dataKey="machine1_energy" name="Machine 1" stroke="#8B5CF6" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="machine2_energy" name="Machine 2" stroke="#06B6D4" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="machine3_energy" name="Machine 3" stroke="#10B981" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="machine4_energy" name="Machine 4" stroke="#F59E0B" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="machine5_energy" name="Machine 5" stroke="#EF4444" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Temperature Chart */}
                  <div className="card p-5" data-testid="machine-temp-chart">
                    <h3 className="font-heading font-semibold text-white mb-4">Machine Temperature Monitoring</h3>
                    <ResponsiveContainer width="100%" height={350}>
                      <AreaChart data={machineData?.data || []}>
                        <defs>
                          <linearGradient id="tempGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#EF4444" stopOpacity={0.2}/>
                            <stop offset="95%" stopColor="#EF4444" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <XAxis dataKey="time" stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                        <YAxis stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" domain={[30, 55]} />
                        <Tooltip content={<CustomTooltip />} />
                        <Area type="monotone" dataKey="machine1_temp" name="Machine 1 Temp" stroke="#EF4444" fill="url(#tempGradient)" strokeWidth={2} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Model Performance View */}
              {activeView === 'performance' && (
                <div className="space-y-6" data-testid="performance-view">
                  {/* Model Training Section */}
                  <div className="card p-5 border-l-4 border-l-success" data-testid="model-training-section">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <Brain className="w-5 h-5 text-success" />
                        <div>
                          <h3 className="font-heading font-semibold text-white">Model Training</h3>
                          <p className="text-zinc-500 text-xs">Train improved prediction models using machine learning</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <select
                          value={selectedAlgorithm}
                          onChange={(e) => setSelectedAlgorithm(e.target.value)}
                          className="bg-surface border border-white/10 rounded-lg px-3 py-2 text-sm text-white font-mono focus:outline-none focus:border-success"
                          data-testid="algorithm-selector"
                        >
                          <option value="gradient_boosting">Gradient Boosting</option>
                          <option value="random_forest">Random Forest</option>
                          <option value="xgboost">XGBoost</option>
                          <option value="linear_regression">Linear Regression</option>
                        </select>
                        <button
                          onClick={() => trainModels('both')}
                          disabled={trainingInProgress}
                          data-testid="train-models-btn"
                          className={`px-4 py-2 rounded-lg font-medium text-sm transition-all flex items-center gap-2
                            ${trainingInProgress 
                              ? 'bg-zinc-700 text-zinc-400 cursor-not-allowed' 
                              : 'bg-success text-white hover:bg-success/80'}`}
                        >
                          {trainingInProgress ? (
                            <>
                              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                              Training...
                            </>
                          ) : (
                            <>
                              <Target className="w-4 h-4" />
                              Train Models
                            </>
                          )}
                        </button>
                        <button
                          onClick={async () => {
                            setComparing(true);
                            try {
                              const [windRes, solarRes] = await Promise.all([
                                axios.post(`${API_URL}/api/compare-models`, { model_type: 'wind', test_size: 0.2 }),
                                axios.post(`${API_URL}/api/compare-models`, { model_type: 'solar', test_size: 0.2 })
                              ]);
                              setComparison({ wind: windRes.data, solar: solarRes.data });
                            } catch (err) { console.error(err); }
                            finally { setComparing(false); }
                          }}
                          disabled={comparing}
                          data-testid="compare-models-btn"
                          className={`px-4 py-2 rounded-lg font-medium text-sm transition-all flex items-center gap-2
                            ${comparing
                              ? 'bg-zinc-700 text-zinc-400 cursor-not-allowed'
                              : 'bg-primary text-white hover:bg-primary/80'}`}
                        >
                          {comparing ? (
                            <>
                              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                              Comparing...
                            </>
                          ) : (
                            <>
                              <BarChart2 className="w-4 h-4" />
                              Compare All
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                    
                    {/* Training Results */}
                    {(modelStatus.wind_trained || modelStatus.solar_trained) && (
                      <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t border-white/10">
                        {modelStatus.wind_metrics && (
                          <div className="bg-wind/5 border border-wind/20 rounded-lg p-4">
                            <div className="flex items-center gap-2 mb-3">
                              <Wind className="w-4 h-4 text-wind" />
                              <span className="text-white font-medium text-sm">Wind Model (Trained)</span>
                              <span className="ml-auto text-xs px-2 py-0.5 bg-success/20 text-success rounded-full">Active</span>
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-xs">
                              <div>
                                <span className="text-zinc-500">Train R²</span>
                                <p className="text-wind font-mono text-lg">{modelStatus.wind_metrics.train_r2}%</p>
                              </div>
                              <div>
                                <span className="text-zinc-500">Test R²</span>
                                <p className="text-wind font-mono text-lg">{modelStatus.wind_metrics.test_r2}%</p>
                              </div>
                              <div>
                                <span className="text-zinc-500">Test MAE</span>
                                <p className="text-white font-mono">{modelStatus.wind_metrics.test_mae} kW</p>
                              </div>
                              <div>
                                <span className="text-zinc-500">Test RMSE</span>
                                <p className="text-white font-mono">{modelStatus.wind_metrics.test_rmse} kW</p>
                              </div>
                            </div>
                          </div>
                        )}
                        {modelStatus.solar_metrics && (
                          <div className="bg-solar/5 border border-solar/20 rounded-lg p-4">
                            <div className="flex items-center gap-2 mb-3">
                              <Sun className="w-4 h-4 text-solar" />
                              <span className="text-white font-medium text-sm">Solar Model (Trained)</span>
                              <span className="ml-auto text-xs px-2 py-0.5 bg-success/20 text-success rounded-full">Active</span>
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-xs">
                              <div>
                                <span className="text-zinc-500">Train R²</span>
                                <p className="text-solar font-mono text-lg">{modelStatus.solar_metrics.train_r2}%</p>
                              </div>
                              <div>
                                <span className="text-zinc-500">Test R²</span>
                                <p className="text-solar font-mono text-lg">{modelStatus.solar_metrics.test_r2}%</p>
                              </div>
                              <div>
                                <span className="text-zinc-500">Test MAE</span>
                                <p className="text-white font-mono">{modelStatus.solar_metrics.test_mae} kW</p>
                              </div>
                              <div>
                                <span className="text-zinc-500">Test RMSE</span>
                                <p className="text-white font-mono">{modelStatus.solar_metrics.test_rmse} kW</p>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Improvement Comparison */}
                    {trainedPredictions && (trainedPredictions.wind_improvement || trainedPredictions.solar_improvement) && (
                      <div className="mt-4 pt-4 border-t border-white/10">
                        <h4 className="text-white font-medium text-sm mb-3">Prediction Improvement for {selectedDate}</h4>
                        <div className="grid grid-cols-2 gap-4">
                          {trainedPredictions.wind_improvement && (
                            <div className="flex items-center gap-4 bg-zinc-800/50 rounded-lg p-3">
                              <Wind className="w-8 h-8 text-wind" />
                              <div>
                                <p className="text-zinc-400 text-xs">Wind MAE Improvement</p>
                                <p className="text-white font-mono">
                                  {trainedPredictions.wind_improvement.original_mae} → {trainedPredictions.wind_improvement.improved_mae} kW
                                </p>
                                <p className={`text-sm font-bold ${trainedPredictions.wind_improvement.improvement_pct > 0 ? 'text-success' : 'text-error'}`}>
                                  {trainedPredictions.wind_improvement.improvement_pct > 0 ? '↑' : '↓'} {Math.abs(trainedPredictions.wind_improvement.improvement_pct)}% improvement
                                </p>
                              </div>
                            </div>
                          )}
                          {trainedPredictions.solar_improvement && (
                            <div className="flex items-center gap-4 bg-zinc-800/50 rounded-lg p-3">
                              <Sun className="w-8 h-8 text-solar" />
                              <div>
                                <p className="text-zinc-400 text-xs">Solar MAE Improvement</p>
                                <p className="text-white font-mono">
                                  {trainedPredictions.solar_improvement.original_mae} → {trainedPredictions.solar_improvement.improved_mae} kW
                                </p>
                                <p className={`text-sm font-bold ${trainedPredictions.solar_improvement.improvement_pct > 0 ? 'text-success' : 'text-error'}`}>
                                  {trainedPredictions.solar_improvement.improvement_pct > 0 ? '↑' : '↓'} {Math.abs(trainedPredictions.solar_improvement.improvement_pct)}% improvement
                                </p>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Model Comparison Results */}
                  {comparison && (
                    <div className="card p-5 border-l-4 border-l-primary" data-testid="model-comparison">
                      <div className="flex items-center gap-3 mb-4">
                        <BarChart2 className="w-5 h-5 text-primary" />
                        <h3 className="font-heading font-semibold text-white">Model Comparison</h3>
                        <span className="px-2 py-1 bg-primary/10 text-primary text-xs rounded-full">4 algorithms</span>
                      </div>
                      <div className="grid grid-cols-2 gap-6">
                        {['wind', 'solar'].map(type => comparison[type] && (
                          <div key={type}>
                            <h4 className={`text-${type} font-medium text-sm mb-3 uppercase tracking-wide`}>
                              {type === 'wind' ? '💨' : '☀️'} {type} — Best: {comparison[type].best_algorithm.replace('_', ' ')}
                            </h4>
                            <table className="w-full text-xs">
                              <thead>
                                <tr className="border-b border-white/10">
                                  <th className="text-left py-2 px-2 text-zinc-400">Algorithm</th>
                                  <th className="text-right py-2 px-2 text-zinc-400">Test R²</th>
                                  <th className="text-right py-2 px-2 text-zinc-400">MAE</th>
                                  <th className="text-right py-2 px-2 text-zinc-400">RMSE</th>
                                  <th className="text-right py-2 px-2 text-zinc-400">Time</th>
                                  <th className="text-right py-2 px-2 text-zinc-400">Overfit</th>
                                </tr>
                              </thead>
                              <tbody>
                                {comparison[type].comparison.map((row, i) => (
                                  <tr key={row.algorithm} className={`border-b border-white/5 ${i === 0 ? 'bg-success/5' : ''}`}>
                                    <td className="py-2 px-2 font-mono text-zinc-300">
                                      {row.algorithm.replace('_', ' ')}
                                      {i === 0 && <span className="ml-1 text-success">★</span>}
                                    </td>
                                    <td className="py-2 px-2 font-mono text-right text-white font-bold">{row.test_r2}%</td>
                                    <td className="py-2 px-2 font-mono text-right text-zinc-300">{row.test_mae}</td>
                                    <td className="py-2 px-2 font-mono text-right text-zinc-300">{row.test_rmse}</td>
                                    <td className="py-2 px-2 font-mono text-right text-zinc-400">{row.train_time_sec}s</td>
                                    <td className={`py-2 px-2 font-mono text-right ${row.overfit_gap > 5 ? 'text-warning' : 'text-success'}`}>
                                      {row.overfit_gap}%
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Improved Predictions Chart */}
                  {trainedPredictions && (trainedPredictions.wind_predictions?.length > 0 || trainedPredictions.solar_predictions?.length > 0) && (
                    <div className="grid grid-cols-2 gap-6">
                      {trainedPredictions.wind_predictions?.length > 0 && (
                        <div className="card p-5" data-testid="improved-wind-chart">
                          <h3 className="font-heading font-semibold text-white mb-2">Wind: Original vs Improved Predictions</h3>
                          <p className="text-zinc-500 text-xs mb-4">Comparing baseline model with trained model</p>
                          <ResponsiveContainer width="100%" height={250}>
                            <LineChart data={trainedPredictions.wind_predictions}>
                              <XAxis dataKey="time" stroke="#52525B" fontSize={10} fontFamily="JetBrains Mono" />
                              <YAxis stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                              <Tooltip content={<CustomTooltip />} />
                              <Line type="monotone" dataKey="actual" name="Actual" stroke="#10B981" strokeWidth={2} dot={false} />
                              <Line type="monotone" dataKey="original_pred" name="Original" stroke="#EF4444" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                              <Line type="monotone" dataKey="improved_pred" name="Improved" stroke="#0EA5E9" strokeWidth={2} dot={false} />
                            </LineChart>
                          </ResponsiveContainer>
                          <div className="flex justify-center gap-4 mt-2 text-xs">
                            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-success" /> Actual</span>
                            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-error border-dashed" /> Original</span>
                            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-wind" /> Improved</span>
                          </div>
                        </div>
                      )}
                      {trainedPredictions.solar_predictions?.length > 0 && (
                        <div className="card p-5" data-testid="improved-solar-chart">
                          <h3 className="font-heading font-semibold text-white mb-2">Solar: Original vs Improved Predictions</h3>
                          <p className="text-zinc-500 text-xs mb-4">Comparing baseline model with trained model</p>
                          <ResponsiveContainer width="100%" height={250}>
                            <LineChart data={trainedPredictions.solar_predictions}>
                              <XAxis dataKey="time" stroke="#52525B" fontSize={10} fontFamily="JetBrains Mono" />
                              <YAxis stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                              <Tooltip content={<CustomTooltip />} />
                              <Line type="monotone" dataKey="actual" name="Actual" stroke="#10B981" strokeWidth={2} dot={false} />
                              <Line type="monotone" dataKey="original_pred" name="Original" stroke="#EF4444" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                              <Line type="monotone" dataKey="improved_pred" name="Improved" stroke="#F59E0B" strokeWidth={2} dot={false} />
                            </LineChart>
                          </ResponsiveContainer>
                          <div className="flex justify-center gap-4 mt-2 text-xs">
                            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-success" /> Actual</span>
                            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-error" /> Original</span>
                            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-solar" /> Improved</span>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Feature Importance Chart */}
                  {featureImportance && (featureImportance.wind || featureImportance.solar) && (
                    <div className="grid grid-cols-2 gap-6" data-testid="feature-importance-section">
                      {featureImportance.wind && (
                        <div className="card p-5 border-l-4 border-l-wind">
                          <div className="flex items-center gap-3 mb-4">
                            <BarChart2 className="w-5 h-5 text-wind" />
                            <h3 className="font-heading font-semibold text-white">Wind — Feature Importance</h3>
                          </div>
                          <ResponsiveContainer width="100%" height={220}>
                            <ComposedChart layout="vertical" data={featureImportance.wind}>
                              <XAxis type="number" stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                              <YAxis type="category" dataKey="feature" stroke="#52525B" fontSize={10} fontFamily="JetBrains Mono" width={160} />
                              <Tooltip content={<CustomTooltip />} />
                              <Bar dataKey="importance" name="Importance" fill="#0EA5E9" radius={[0, 4, 4, 0]} />
                            </ComposedChart>
                          </ResponsiveContainer>
                        </div>
                      )}
                      {featureImportance.solar && (
                        <div className="card p-5 border-l-4 border-l-solar">
                          <div className="flex items-center gap-3 mb-4">
                            <BarChart2 className="w-5 h-5 text-solar" />
                            <h3 className="font-heading font-semibold text-white">Solar — Feature Importance</h3>
                          </div>
                          <ResponsiveContainer width="100%" height={220}>
                            <ComposedChart layout="vertical" data={featureImportance.solar}>
                              <XAxis type="number" stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                              <YAxis type="category" dataKey="feature" stroke="#52525B" fontSize={10} fontFamily="JetBrains Mono" width={160} />
                              <Tooltip content={<CustomTooltip />} />
                              <Bar dataKey="importance" name="Importance" fill="#F59E0B" radius={[0, 4, 4, 0]} />
                            </ComposedChart>
                          </ResponsiveContainer>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Original Metrics Summary */}
                  <div className="grid grid-cols-2 gap-6">
                    {/* Wind Model Metrics */}
                    <div className="card p-5 border-l-4 border-l-wind">
                      <div className="flex items-center gap-3 mb-4">
                        <Wind className="w-5 h-5 text-wind" />
                        <h3 className="font-heading font-semibold text-white">Wind Model Performance</h3>
                        <span className="text-xs px-2 py-0.5 bg-zinc-700 text-zinc-400 rounded-full">Original</span>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <MetricCard 
                          label="MAPE Score" 
                          value={performanceData?.wind_overall?.avg_mape_score || '--'} 
                          unit="%" 
                          color="wind"
                          icon={Target}
                        />
                        <MetricCard 
                          label="RMSE" 
                          value={performanceData?.wind_overall?.avg_rmse || '--'} 
                          unit="kW" 
                          subtext="Root Mean Square Error"
                          color="wind"
                          icon={Activity}
                        />
                        <MetricCard 
                          label="MAE" 
                          value={performanceData?.wind_overall?.avg_mae || '--'} 
                          unit="kW" 
                          subtext="Mean Absolute Error"
                          color="wind"
                          icon={BarChart2}
                        />
                        <MetricCard 
                          label="MAPE" 
                          value={performanceData?.wind_overall?.avg_mape || '--'} 
                          unit="%" 
                          subtext="Mean Abs % Error"
                          color="wind"
                          icon={TrendingUp}
                        />
                      </div>
                      <div className="mt-4 pt-4 border-t border-white/10 flex justify-between text-xs">
                        <span className="text-zinc-500">Best: <span className="text-success">{performanceData?.wind_overall?.best_day || '--'}</span></span>
                        <span className="text-zinc-500">Worst: <span className="text-error">{performanceData?.wind_overall?.worst_day || '--'}</span></span>
                      </div>
                    </div>

                    {/* Solar Model Metrics */}
                    <div className="card p-5 border-l-4 border-l-solar">
                      <div className="flex items-center gap-3 mb-4">
                        <Sun className="w-5 h-5 text-solar" />
                        <h3 className="font-heading font-semibold text-white">Solar Model Performance</h3>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <MetricCard 
                          label="MAPE Score" 
                          value={performanceData?.solar_overall?.avg_mape_score || '--'} 
                          unit="%" 
                          color="solar"
                          icon={Target}
                        />
                        <MetricCard 
                          label="RMSE" 
                          value={performanceData?.solar_overall?.avg_rmse || '--'} 
                          unit="kW" 
                          subtext="Root Mean Square Error"
                          color="solar"
                          icon={Activity}
                        />
                        <MetricCard 
                          label="MAE" 
                          value={performanceData?.solar_overall?.avg_mae || '--'} 
                          unit="kW" 
                          subtext="Mean Absolute Error"
                          color="solar"
                          icon={BarChart2}
                        />
                        <MetricCard 
                          label="MAPE" 
                          value={performanceData?.solar_overall?.avg_mape || '--'} 
                          unit="%" 
                          subtext="Mean Abs % Error"
                          color="solar"
                          icon={TrendingUp}
                        />
                      </div>
                      <div className="mt-4 pt-4 border-t border-white/10 flex justify-between text-xs">
                        <span className="text-zinc-500">Best: <span className="text-success">{performanceData?.solar_overall?.best_day || '--'}</span></span>
                        <span className="text-zinc-500">Worst: <span className="text-error">{performanceData?.solar_overall?.worst_day || '--'}</span></span>
                      </div>
                    </div>
                  </div>

                  {/* MAPE Score Trend Chart */}
                  <div className="card p-5" data-testid="accuracy-trend-chart">
                    <h3 className="font-heading font-semibold text-white mb-4">Prediction Score Trend — 100 − MAPE% (30 Days)</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={performanceData?.wind_trend || []}>
                        <XAxis dataKey="date" stroke="#52525B" fontSize={10} fontFamily="JetBrains Mono" angle={-45} textAnchor="end" height={60} />
                        <YAxis stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" domain={[0, 100]} />
                        <Tooltip content={<CustomTooltip />} />
                        <ReferenceLine y={80} stroke="#10B981" strokeDasharray="3 3" label={{ value: 'Target 80%', fill: '#10B981', fontSize: 10 }} />
                        <Line type="monotone" dataKey="mape_score" name="Wind MAPE Score" stroke="#0EA5E9" strokeWidth={2} dot={{ r: 3 }} />
                      </LineChart>
                    </ResponsiveContainer>
                    <div className="mt-4">
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={performanceData?.solar_trend || []}>
                          <XAxis dataKey="date" stroke="#52525B" fontSize={10} fontFamily="JetBrains Mono" angle={-45} textAnchor="end" height={60} />
                          <YAxis stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" domain={[0, 100]} />
                          <Tooltip content={<CustomTooltip />} />
                          <ReferenceLine y={80} stroke="#10B981" strokeDasharray="3 3" label={{ value: 'Target 80%', fill: '#10B981', fontSize: 10 }} />
                          <Line type="monotone" dataKey="mape_score" name="Solar MAPE Score" stroke="#F59E0B" strokeWidth={2} dot={{ r: 3 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Residual Analysis */}
                  <div className="grid grid-cols-2 gap-6">
                    <div className="card p-5" data-testid="wind-residuals-chart">
                      <h3 className="font-heading font-semibold text-white mb-2">Wind Model Residuals</h3>
                      <p className="text-zinc-500 text-xs mb-4">Prediction errors by hour for {performanceData?.selected_date}</p>
                      <ResponsiveContainer width="100%" height={250}>
                        <ComposedChart data={performanceData?.wind_residuals || []}>
                          <XAxis dataKey="time" stroke="#52525B" fontSize={10} fontFamily="JetBrains Mono" />
                          <YAxis stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                          <Tooltip content={<CustomTooltip />} />
                          <ReferenceLine y={0} stroke="#52525B" />
                          <Bar dataKey="residual" name="Residual" fill="#0EA5E9" opacity={0.7} />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="card p-5" data-testid="solar-residuals-chart">
                      <h3 className="font-heading font-semibold text-white mb-2">Solar Model Residuals</h3>
                      <p className="text-zinc-500 text-xs mb-4">Prediction errors by hour for {performanceData?.selected_date}</p>
                      <ResponsiveContainer width="100%" height={250}>
                        <ComposedChart data={performanceData?.solar_residuals || []}>
                          <XAxis dataKey="time" stroke="#52525B" fontSize={10} fontFamily="JetBrains Mono" />
                          <YAxis stroke="#52525B" fontSize={11} fontFamily="JetBrains Mono" />
                          <Tooltip content={<CustomTooltip />} />
                          <ReferenceLine y={0} stroke="#52525B" />
                          <Bar dataKey="residual" name="Residual" fill="#F59E0B" opacity={0.7} />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Error Distribution Table */}
                  <div className="card p-5" data-testid="error-distribution">
                    <h3 className="font-heading font-semibold text-white mb-4">Hourly Error Analysis ({performanceData?.selected_date})</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-white/10">
                            <th className="text-left py-3 px-4 text-zinc-400 font-medium">Hour</th>
                            <th className="text-right py-3 px-4 text-wind font-medium">Wind Actual</th>
                            <th className="text-right py-3 px-4 text-wind font-medium">Wind Pred</th>
                            <th className="text-right py-3 px-4 text-wind font-medium">Wind Error %</th>
                            <th className="text-right py-3 px-4 text-solar font-medium">Solar Actual</th>
                            <th className="text-right py-3 px-4 text-solar font-medium">Solar Pred</th>
                            <th className="text-right py-3 px-4 text-solar font-medium">Solar Error %</th>
                          </tr>
                        </thead>
                        <tbody>
                          {performanceData?.wind_residuals?.slice(0, 12).map((wind, i) => {
                            const solar = performanceData?.solar_residuals?.[i] || {};
                            return (
                              <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                                <td className="py-2 px-4 font-mono text-zinc-300">{wind.time}</td>
                                <td className="py-2 px-4 font-mono text-right">{wind.actual}</td>
                                <td className="py-2 px-4 font-mono text-right">{wind.predicted}</td>
                                <td className={`py-2 px-4 font-mono text-right ${wind.error_pct > 30 ? 'text-error' : wind.error_pct > 15 ? 'text-warning' : 'text-success'}`}>
                                  {wind.error_pct}%
                                </td>
                                <td className="py-2 px-4 font-mono text-right">{solar.actual || '--'}</td>
                                <td className="py-2 px-4 font-mono text-right">{solar.predicted || '--'}</td>
                                <td className={`py-2 px-4 font-mono text-right ${(solar.error_pct || 0) > 30 ? 'text-error' : (solar.error_pct || 0) > 15 ? 'text-warning' : 'text-success'}`}>
                                  {solar.error_pct || '--'}%
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Experiment History */}
                  {experiments.length > 0 && (
                    <div className="card p-5" data-testid="experiment-history">
                      <div className="flex items-center gap-3 mb-4">
                        <Activity className="w-5 h-5 text-primary" />
                        <h3 className="font-heading font-semibold text-white">Experiment History</h3>
                        <span className="px-2 py-1 bg-primary/10 text-primary text-xs rounded-full">
                          {experiments.length} runs
                        </span>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b border-white/10">
                              <th className="text-left py-3 px-3 text-zinc-400 font-medium">#</th>
                              <th className="text-left py-3 px-3 text-zinc-400 font-medium">Timestamp</th>
                              <th className="text-left py-3 px-3 text-zinc-400 font-medium">Algorithm</th>
                              <th className="text-left py-3 px-3 text-zinc-400 font-medium">Type</th>
                              <th className="text-right py-3 px-3 text-zinc-400 font-medium">Test Size</th>
                              <th className="text-right py-3 px-3 text-wind font-medium">Wind R²</th>
                              <th className="text-right py-3 px-3 text-wind font-medium">Wind MAE</th>
                              <th className="text-right py-3 px-3 text-solar font-medium">Solar R²</th>
                              <th className="text-right py-3 px-3 text-solar font-medium">Solar MAE</th>
                            </tr>
                          </thead>
                          <tbody>
                            {experiments.slice().reverse().map((exp) => (
                              <tr key={exp.id} className="border-b border-white/5 hover:bg-white/5">
                                <td className="py-2 px-3 font-mono text-zinc-300">{exp.id}</td>
                                <td className="py-2 px-3 font-mono text-zinc-400 text-xs">{exp.timestamp?.split('T')[0]} {exp.timestamp?.split('T')[1]?.split('.')[0]}</td>
                                <td className="py-2 px-3">
                                  <span className="px-2 py-0.5 bg-zinc-800 rounded text-xs text-zinc-300">
                                    {exp.algorithm?.replace('_', ' ')}
                                  </span>
                                </td>
                                <td className="py-2 px-3 text-zinc-300">{exp.model_type}</td>
                                <td className="py-2 px-3 font-mono text-right text-zinc-300">{(exp.test_size * 100).toFixed(0)}%</td>
                                <td className="py-2 px-3 font-mono text-right text-wind">{exp.results?.wind?.test_r2 ?? '--'}%</td>
                                <td className="py-2 px-3 font-mono text-right text-wind">{exp.results?.wind?.test_mae ?? '--'}</td>
                                <td className="py-2 px-3 font-mono text-right text-solar">{exp.results?.solar?.test_r2 ?? '--'}%</td>
                                <td className="py-2 px-3 font-mono text-right text-solar">{exp.results?.solar?.test_mae ?? '--'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
