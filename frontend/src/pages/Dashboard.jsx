import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Package, 
  Users, 
  Truck, 
  Train, 
  DollarSign, 
  MapPin,
  TrendingUp,
  AlertTriangle,
  Zap
} from 'lucide-react';
import StatCard from '../components/StatCard';
import { apiService } from '../services/api';
import toast from 'react-hot-toast';

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardStats();
  }, []);

  const loadDashboardStats = async () => {
    try {
      const response = await apiService.getDashboardStats();
      setStats(response.data);
    } catch (error) {
      toast.error('Failed to load dashboard statistics');
      console.error('Error loading stats:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-8"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-200 rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-2">
          Overview of rake formation optimization system
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Total Orders"
          value={stats?.total_orders || 0}
          subtitle="Active plans"
          icon={Package}
          color="blue"
        />
        <StatCard
          title="Total Tonnage"
          value={`${(stats?.total_tonnage / 1000).toFixed(0)}K`}
          subtitle="Tons"
          icon={TrendingUp}
          color="green"
        />
        <StatCard
          title="Customers"
          value={stats?.unique_customers || 0}
          subtitle="Unique clients"
          icon={Users}
          color="orange"
        />
        <StatCard
          title="Avg Priority"
          value={stats?.avg_priority || 0}
          subtitle="Score"
          icon={AlertTriangle}
          color="purple"
        />
      </div>

      {/* Additional Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <StatCard
          title="Origins"
          value={stats?.unique_origins || 0}
          subtitle="Steel plants"
          icon={MapPin}
          color="blue"
        />
        <StatCard
          title="Destinations"
          value={stats?.unique_destinations || 0}
          subtitle="Delivery points"
          icon={Truck}
          color="green"
        />
        <StatCard
          title="Avg Distance"
          value={`${stats?.avg_distance || 0} km`}
          subtitle="Transportation"
          icon={Train}
          color="orange"
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Quick Actions
          </h3>
          <div className="space-y-3">
            <Link
              to="/optimize"
              className="w-full btn-primary flex items-center justify-center"
            >
              <Zap className="h-4 w-4 mr-2" />
              Start New Optimization
            </Link>
            <Link
              to="/plans"
              className="w-full btn-secondary flex items-center justify-center"
            >
              <Package className="h-4 w-4 mr-2" />
              View All Plans
            </Link>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            System Status
          </h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between py-2">
              <span className="text-gray-600">ML Pipeline</span>
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                Operational
              </span>
            </div>
            <div className="flex items-center justify-between py-2">
              <span className="text-gray-600">Optimization Engine</span>
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                Ready
              </span>
            </div>
            <div className="flex items-center justify-between py-2">
              <span className="text-gray-600">Data Source</span>
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                Connected
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;