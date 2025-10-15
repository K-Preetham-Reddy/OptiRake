import React from 'react';
import { Loader, CheckCircle, XCircle, Clock } from 'lucide-react';

const OptimizationProgress = ({ jobStatus }) => {
  const { status, progress, message } = jobStatus;
  
  const statusConfig = {
    queued: { color: 'text-yellow-600', icon: Clock, bg: 'bg-yellow-50' },
    running: { color: 'text-blue-600', icon: Loader, bg: 'bg-blue-50' },
    completed: { color: 'text-green-600', icon: CheckCircle, bg: 'bg-green-50' },
    failed: { color: 'text-red-600', icon: XCircle, bg: 'bg-red-50' },
  };
  
  const config = statusConfig[status] || statusConfig.queued;
  const Icon = config.icon;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Optimization Progress</h3>
        <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${config.bg} ${config.color}`}>
          <Icon className="h-4 w-4 mr-1" />
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </div>
      </div>
      
      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-sm text-gray-600 mb-1">
            <span>Progress</span>
            <span>{Math.round(progress * 100)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-primary-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress * 100}%` }}
            ></div>
          </div>
        </div>
        
        <div className="text-sm text-gray-600">
          <p>{message}</p>
        </div>
        
        {status === 'running' && (
          <div className="flex items-center text-sm text-blue-600">
            <Loader className="h-4 w-4 mr-2 animate-spin" />
            Processing... This may take a few minutes
          </div>
        )}
      </div>
    </div>
  );
};

export default OptimizationProgress;