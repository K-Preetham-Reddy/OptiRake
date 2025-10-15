import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Zap, Package, CheckCircle, AlertCircle, Play } from 'lucide-react';
import OptimizationProgress from '../components/OptimizationProgress';
import { apiService } from '../services/api';
import toast from 'react-hot-toast';

const Optimization = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [selectedPlans, setSelectedPlans] = useState([]);
  const [availablePlans, setAvailablePlans] = useState([]);
  const [optimizationMode, setOptimizationMode] = useState('balanced');
  const [jobStatus, setJobStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadAvailablePlans();
    const planId = searchParams.get('plan');
    if (planId) {
      setSelectedPlans([planId]);
    }
  }, [searchParams]);

  const loadAvailablePlans = async () => {
    try {
      const response = await apiService.getPlans({ limit: 100 });
      setAvailablePlans(response.data.plans);
    } catch (error) {
      toast.error('Failed to load plans');
      console.error('Error loading plans:', error);
    }
  };

  const togglePlanSelection = (planId) => {
    setSelectedPlans(prev =>
      prev.includes(planId)
        ? prev.filter(id => id !== planId)
        : [...prev, planId]
    );
  };

  const selectAllPlans = () => {
    setSelectedPlans(availablePlans.map(plan => plan.plan_id));
  };

  const clearSelection = () => {
    setSelectedPlans([]);
  };

  const startOptimization = async () => {
    if (selectedPlans.length === 0) {
      toast.error('Please select at least one plan to optimize');
      return;
    }

    setLoading(true);
    try {
      const response = await apiService.startOptimization({
        plan_ids: selectedPlans,
        optimization_mode: optimizationMode
      });

      const jobId = response.data.job_id;
      setJobStatus({
        jobId,
        status: 'queued',
        progress: 0,
        message: 'Job queued for processing'
      });

      // Poll for job status
      pollJobStatus(jobId);
    } catch (error) {
      toast.error('Failed to start optimization');
      console.error('Error starting optimization:', error);
      setLoading(false);
    }
  };

  const pollJobStatus = async (jobId) => {
    const poll = async () => {
      try {
        const response = await apiService.getJobStatus(jobId);
        const status = response.data;
        
        setJobStatus({
          jobId: status.job_id,
          status: status.status,
          progress: status.progress,
          message: status.message
        });

        if (status.status === 'completed') {
          toast.success('Optimization completed successfully!');
          navigate(`/results/${jobId}`);
        } else if (status.status === 'failed') {
          toast.error('Optimization failed');
          setLoading(false);
        } else if (status.status === 'running' || status.status === 'queued') {
          setTimeout(poll, 2000); // Continue polling
        }
      } catch (error) {
        console.error('Error polling job status:', error);
        setLoading(false);
      }
    };

    poll();
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Optimize Rake Formation</h1>
        <p className="text-gray-600 mt-2">
          Use AI-powered optimization to determine the most efficient transportation plan
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column - Configuration */}
        <div className="lg:col-span-2 space-y-6">
          {/* Optimization Mode */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Zap className="h-5 w-5 mr-2" />
              Optimization Strategy
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                { id: 'cost_min', name: 'Cost Minimization', description: 'Focus on reducing total transportation costs' },
                { id: 'balanced', name: 'Balanced Approach', description: 'Balance cost, reliability, and rail utilization' },
                { id: 'rail_max', name: 'Maximize Rail', description: 'Prioritize rail transportation where feasible' }
              ].map((mode) => (
                <label key={mode.id} className="relative cursor-pointer">
                  <input
                    type="radio"
                    name="optimizationMode"
                    value={mode.id}
                    checked={optimizationMode === mode.id}
                    onChange={(e) => setOptimizationMode(e.target.value)}
                    className="sr-only"
                  />
                  <div className={`border-2 rounded-lg p-4 transition-all duration-200 ${
                    optimizationMode === mode.id
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}>
                    <div className="flex items-center mb-2">
                      <div className={`w-4 h-4 rounded-full border-2 mr-2 ${
                        optimizationMode === mode.id
                          ? 'border-primary-500 bg-primary-500'
                          : 'border-gray-300'
                      }`}></div>
                      <span className="font-semibold text-gray-900">{mode.name}</span>
                    </div>
                    <p className="text-sm text-gray-600">{mode.description}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Plan Selection */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                <Package className="h-5 w-5 mr-2" />
                Select Plans for Optimization
              </h3>
              <div className="flex space-x-2">
                <button
                  onClick={selectAllPlans}
                  className="btn-secondary text-sm"
                >
                  Select All
                </button>
                <button
                  onClick={clearSelection}
                  className="btn-secondary text-sm"
                >
                  Clear
                </button>
              </div>
            </div>

            <div className="space-y-3 max-h-96 overflow-y-auto">
              {availablePlans.map((plan) => (
                <label key={plan.plan_id} className="flex items-center p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors duration-200">
                  <input
                    type="checkbox"
                    checked={selectedPlans.includes(plan.plan_id)}
                    onChange={() => togglePlanSelection(plan.plan_id)}
                    className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                  />
                  <div className="ml-3 flex-1">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-gray-900">{plan.plan_id}</span>
                      <span className="text-sm text-gray-500">{plan.planned_qty_t} tons</span>
                    </div>
                    <div className="flex items-center justify-between text-sm text-gray-600">
                      <span>{plan.customer_name}</span>
                      <span>{plan.origin_plant} → {plan.destination}</span>
                    </div>
                  </div>
                </label>
              ))}
            </div>

            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">
                  {selectedPlans.length} of {availablePlans.length} plans selected
                </span>
                <span className="font-semibold">
                  Total: {availablePlans
                    .filter(plan => selectedPlans.includes(plan.plan_id))
                    .reduce((sum, plan) => sum + plan.planned_qty_t, 0)} tons
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column - Actions & Progress */}
        <div className="space-y-6">
          {/* Start Optimization */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Ready to Optimize</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Selected Plans</span>
                <span className="font-semibold">{selectedPlans.length}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Total Tonnage</span>
                <span className="font-semibold">
                  {availablePlans
                    .filter(plan => selectedPlans.includes(plan.plan_id))
                    .reduce((sum, plan) => sum + plan.planned_qty_t, 0)} tons
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Strategy</span>
                <span className="font-semibold capitalize">
                  {optimizationMode.replace('_', ' ')}
                </span>
              </div>

              <button
                onClick={startOptimization}
                disabled={loading || selectedPlans.length === 0}
                className="w-full btn-primary flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Play className="h-4 w-4 mr-2" />
                {loading ? 'Starting...' : 'Start Optimization'}
              </button>

              {selectedPlans.length === 0 && (
                <div className="flex items-center text-sm text-amber-600 bg-amber-50 p-3 rounded-lg">
                  <AlertCircle className="h-4 w-4 mr-2" />
                  Select at least one plan to optimize
                </div>
              )}
            </div>
          </div>

          {/* Progress */}
          {jobStatus && (
            <OptimizationProgress jobStatus={jobStatus} />
          )}

          {/* Tips */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Optimization Tips</h3>
            <div className="space-y-3 text-sm text-gray-600">
              <div className="flex items-start">
                <CheckCircle className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                <span>Select multiple plans for better optimization results</span>
              </div>
              <div className="flex items-start">
                <CheckCircle className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                <span>Consider selecting plans from the same origin for efficient rake formation</span>
              </div>
              <div className="flex items-start">
                <CheckCircle className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                <span>Higher priority plans will be prioritized in the optimization</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Optimization;