import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Search, Filter, Package, MapPin, Calendar, ArrowUpDown } from 'lucide-react';
import { apiService } from '../services/api';
import toast from 'react-hot-toast';

const Plans = () => {
  const [plans, setPlans] = useState([]);
  const [filteredPlans, setFilteredPlans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    origin: '',
    destination: '',
    search: ''
  });
  const [filterOptions, setFilterOptions] = useState({});

  useEffect(() => {
    loadPlans();
    loadFilterOptions();
  }, []);

  useEffect(() => {
    filterPlans();
  }, [plans, filters]);

  const loadPlans = async () => {
    try {
      const response = await apiService.getPlans();
      setPlans(response.data.plans);
    } catch (error) {
      toast.error('Failed to load plans');
      console.error('Error loading plans:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadFilterOptions = async () => {
    try {
      const response = await apiService.getFilterOptions();
      setFilterOptions(response.data);
    } catch (error) {
      console.error('Error loading filter options:', error);
    }
  };

  const filterPlans = () => {
    let filtered = plans;

    if (filters.origin) {
      filtered = filtered.filter(plan => plan.origin_plant === filters.origin);
    }

    if (filters.destination) {
      filtered = filtered.filter(plan => plan.destination === filters.destination);
    }

    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      filtered = filtered.filter(plan =>
        plan.customer_name.toLowerCase().includes(searchLower) ||
        plan.plan_id.toLowerCase().includes(searchLower) ||
        plan.product_type.toLowerCase().includes(searchLower)
      );
    }

    setFilteredPlans(filtered);
  };

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
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

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Transportation Plans</h1>
        <p className="text-gray-600 mt-2">
          Manage and review all transportation plans
        </p>
      </div>

      {/* Filters */}
      <div className="card mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <Filter className="h-5 w-5 mr-2" />
            Filters
          </h3>
          <div className="text-sm text-gray-500">
            {filteredPlans.length} of {plans.length} plans
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Search
            </label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search plans..."
                className="input-field pl-10"
                value={filters.search}
                onChange={(e) => handleFilterChange('search', e.target.value)}
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Origin
            </label>
            <select
              className="input-field"
              value={filters.origin}
              onChange={(e) => handleFilterChange('origin', e.target.value)}
            >
              <option value="">All Origins</option>
              {filterOptions.origins?.map(origin => (
                <option key={origin} value={origin}>{origin}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Destination
            </label>
            <select
              className="input-field"
              value={filters.destination}
              onChange={(e) => handleFilterChange('destination', e.target.value)}
            >
              <option value="">All Destinations</option>
              {filterOptions.destinations?.map(dest => (
                <option key={dest} value={dest}>{dest}</option>
              ))}
            </select>
          </div>

          <div className="flex items-end">
            <button
              onClick={() => setFilters({ origin: '', destination: '', search: '' })}
              className="btn-secondary w-full"
            >
              Clear Filters
            </button>
          </div>
        </div>
      </div>

      {/* Plans List */}
      <div className="space-y-4">
        {filteredPlans.map((plan) => (
          <div key={plan.plan_id} className="card hover:shadow-md transition-shadow duration-200">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="bg-primary-50 p-3 rounded-lg">
                  <Package className="h-6 w-6 text-primary-600" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">{plan.plan_id}</h4>
                  <p className="text-gray-600">{plan.customer_name}</p>
                </div>
              </div>

              <div className="flex items-center space-x-6 text-sm">
                <div className="text-center">
                  <p className="text-gray-500">Quantity</p>
                  <p className="font-semibold">{plan.planned_qty_t} tons</p>
                </div>

                <div className="text-center">
                  <p className="text-gray-500">Priority</p>
                  <p className="font-semibold">{plan.priority_score}</p>
                </div>

                <div className="flex items-center space-x-2 text-gray-500">
                  <MapPin className="h-4 w-4" />
                  <span>{plan.origin_plant} → {plan.destination}</span>
                </div>

                <div className="flex items-center space-x-2 text-gray-500">
                  <Calendar className="h-4 w-4" />
                  <span>{plan.plan_date}</span>
                </div>

                <Link
                  to={`/optimize?plan=${plan.plan_id}`}
                  className="btn-primary text-sm"
                >
                  Optimize
                </Link>
              </div>
            </div>
          </div>
        ))}

        {filteredPlans.length === 0 && (
          <div className="card text-center py-12">
            <Package className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">No plans found</h3>
            <p className="text-gray-600">
              Try adjusting your filters to see more results.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Plans;