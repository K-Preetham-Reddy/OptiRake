import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { 
  ArrowLeft, 
  Download, 
  Train, 
  Truck, 
  DollarSign, 
  Package,
  BarChart3,
  MapPin
} from 'lucide-react';
import { apiService } from '../services/api';
import toast from 'react-hot-toast';

const Results = () => {
  const { jobId } = useParams();
  const [results, setResults] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (jobId) {
      loadResults();
    }
  }, [jobId]);

  const loadResults = async () => {
    try {
      const response = await apiService.getResults(jobId, { limit: 100 });
      setResults(response.data);
      setSummary(response.data.summary);
    } catch (error) {
      toast.error('Failed to load optimization results');
      console.error('Error loading results:', error);
    } finally {
      setLoading(false);
    }
  };

  const exportToCSV = () => {
    // Simple CSV export implementation
    const headers = ['Plan ID', 'Mode', 'Rail Tons', 'Total Cost', 'Customer', 'Origin', 'Destination'];
    const csvData = results.results.map(row => [
      row.plan_id,
      row.optimized_mode,
      row.q_rail_tons,
      row.optimized_total_cost,
      row.customer_name,
      row.origin_plant,
      row.destination
    ]);

    const csvContent = [
      headers.join(','),
      ...csvData.map(row => row.join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `optimization_results_${jobId}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-8"></div>
          <div className="grid grid-cols-1 gap-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-20 bg-gray-200 rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">Results Not Found</h1>
          <p className="text-gray-600 mb-8">The requested optimization results could not be found.</p>
          <Link to="/optimize" className="btn-primary">
            Start New Optimization
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <Link to="/optimize" className="inline-flex items-center text-primary-600 hover:text-primary-700 mb-2">
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back to Optimization
          </Link>
          <h1 className="text-3xl font-bold text-gray-900">Optimization Results</h1>
          <p className="text-gray-600 mt-2">
            Job ID: {jobId} • {summary?.optimization_mode && `${summary.optimization_mode.replace('_', ' ')} strategy`}
          </p>
        </div>
        <button
          onClick={exportToCSV}
          className="btn-secondary flex items-center"
        >
          <Download className="h-4 w-4 mr-2" />
          Export CSV
        </button>
      </div>

      {/* Summary Cards */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="card bg-gradient-to-br from-blue-50 to-blue-100">
            <div className="flex items-center">
              <div className="bg-blue-100 p-3 rounded-lg">
                <Package className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-blue-600">Total Orders</p>
                <p className="text-2xl font-semibold text-blue-900">{summary.total_orders}</p>
              </div>
            </div>
          </div>

          <div className="card bg-gradient-to-br from-green-50 to-green-100">
            <div className="flex items-center">
              <div className="bg-green-100 p-3 rounded-lg">
                <Train className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-green-600">Rail Orders</p>
                <p className="text-2xl font-semibold text-green-900">
                  {summary.rail_orders} ({summary.rail_orders_percentage}%)
                </p>
              </div>
            </div>
          </div>

          <div className="card bg-gradient-to-br from-orange-50 to-orange-100">
            <div className="flex items-center">
              <div className="bg-orange-100 p-3 rounded-lg">
                <Truck className="h-6 w-6 text-orange-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-orange-600">Road Orders</p>
                <p className="text-2xl font-semibold text-orange-900">
                  {summary.road_orders} ({100 - summary.rail_orders_percentage}%)
                </p>
              </div>
            </div>
          </div>

          <div className="card bg-gradient-to-br from-purple-50 to-purple-100">
            <div className="flex items-center">
              <div className="bg-purple-100 p-3 rounded-lg">
                <DollarSign className="h-6 w-6 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-purple-600">Total Cost</p>
                <p className="text-2xl font-semibold text-purple-900">
                  ₹{summary.total_cost?.toLocaleString()}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Detailed Results */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <BarChart3 className="h-5 w-5 mr-2" />
            Optimization Results
          </h3>
          <div className="text-sm text-gray-500">
            Showing {results.pagination.returned} of {results.pagination.total} results
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-900">Plan ID</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-900">Customer</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-900">Route</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-900">Mode</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-900">Rail Tons</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-900">Total Cost</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-900">On-Time Probability</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {results.results.map((result, index) => (
                <tr key={index} className="hover:bg-gray-50 transition-colors duration-200">
                  <td className="py-3 px-4 text-sm font-medium text-gray-900">
                    {result.plan_id}
                  </td>
                  <td className="py-3 px-4 text-sm text-gray-600">
                    {result.customer_name}
                  </td>
                  <td className="py-3 px-4 text-sm text-gray-600">
                    <div className="flex items-center">
                      <MapPin className="h-3 w-3 mr-1 text-gray-400" />
                      {result.origin_plant} → {result.destination}
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      result.optimized_mode === 'Rail' 
                        ? 'bg-green-100 text-green-800'
                        : 'bg-orange-100 text-orange-800'
                    }`}>
                      {result.optimized_mode === 'Rail' ? (
                        <Train className="h-3 w-3 mr-1" />
                      ) : (
                        <Truck className="h-3 w-3 mr-1" />
                      )}
                      {result.optimized_mode}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-sm text-gray-600">
                    {result.q_rail_tons > 0 ? `${result.q_rail_tons} tons` : '-'}
                  </td>
                  <td className="py-3 px-4 text-sm font-medium text-gray-900">
                    ₹{result.optimized_total_cost?.toLocaleString()}
                  </td>
                  <td className="py-3 px-4 text-sm text-gray-600">
                    {(result.on_time_prob * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {results.results.length === 0 && (
          <div className="text-center py-12">
            <Package className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">No results found</h3>
            <p className="text-gray-600">
              No optimization results are available for this job.
            </p>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex justify-center space-x-4 mt-8">
        <Link to="/optimize" className="btn-primary">
          Run New Optimization
        </Link>
        <Link to="/plans" className="btn-secondary">
          View All Plans
        </Link>
      </div>
    </div>
  );
};

export default Results;