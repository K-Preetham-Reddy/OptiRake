import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Train, BarChart3, List, Settings, Zap } from 'lucide-react';

const Navigation = () => {
  const location = useLocation();
  
  const navigation = [
    { name: 'Dashboard', href: '/', icon: BarChart3 },
    { name: 'Plans', href: '/plans', icon: List },
    { name: 'Optimize', href: '/optimize', icon: Zap },
  ];
  
  const isCurrentPath = (path) => location.pathname === path;

  return (
    <nav className="fixed top-0 left-0 right-0 bg-white shadow-sm border-b border-gray-200 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <Train className="h-8 w-8 text-primary-600" />
              <span className="ml-2 text-xl font-bold text-steel-900">
                Rake Formation System
              </span>
            </div>
            <div className="hidden md:ml-8 md:flex md:space-x-4">
              {navigation.map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`inline-flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                      isCurrentPath(item.href)
                        ? 'bg-primary-100 text-primary-700'
                        : 'text-steel-600 hover:text-steel-900 hover:bg-steel-50'
                    }`}
                  >
                    <Icon className="h-4 w-4 mr-2" />
                    {item.name}
                  </Link>
                );
              })}
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button className="p-2 text-steel-600 hover:text-steel-900 transition-colors duration-200">
              <Settings className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;