import { Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { useAppDispatch, useAppSelector } from '../redux/hooks';
import { Navigate } from 'react-router-dom';
import { login, setLoading } from '../redux/slices/authSlice';
import api from '../api/axios';
import Sidebar from './Sidebar';
import SidebarToggle from './SiderbarToggle';
import MainContent from './MainContent';
import ReportPage from './ReportPage';
import Login from './Login';
import Register from './Register';
import NotePage from './NotePage';
import MemoryPage from './MemoryPage';
import ProtectedRoute from './ProtectedRoute';
import styles from '../styles/Main.module.scss'
import LandingPage from '../pages/Landing/LandingPage'
import AgentDemoPage from '../pages/AgentDemo/AgentDemoPage'



const AppContent = () => {
    const dispatch = useAppDispatch()
    const location = useLocation()
    const navigate = useNavigate()
    const { isAuthenticated } = useAppSelector((state) => state.auth)
    const [isSidebarOpen, setIsSidebarOpen] = useState(false)

    useEffect(() => {
        let isMounted = true;

        const attemptRefresh = async () => {

            if (isAuthenticated) {
                if (isMounted) dispatch(setLoading(false))
                return;
            }

            try {
                const { data } = await api.post('/auth/refresh');
                if (isMounted) {
                    dispatch(login({ accessToken: data.accessToken, user: data.user }));
                }
            } catch (err) {
                // Refresh failed — user is not logged in.
                // Don't navigate here. ProtectedRoute will handle redirects
                // for protected pages; public pages (/, /login, /register) stay accessible.
                console.log('Refresh failed:', err);
            } finally {
                if (isMounted) {
                    dispatch(setLoading(false))
                }
            }
        };

        attemptRefresh();

        return () => {
            isMounted = false;
        };
    }, []);

    const hideSidebar = !isAuthenticated ||
        location.pathname === '/login' ||
        location.pathname === '/register' ||
        location.pathname === "/"


    const toggleSidebar = () => {
        setIsSidebarOpen(!isSidebarOpen)
    }

    const closeSidebar = () => {
        setIsSidebarOpen(false)
    }

    return (
        <div className={styles.appContainer}>
            {!hideSidebar && (
                <>
                    <SidebarToggle
                        isOpen={isSidebarOpen}
                        onToggle={toggleSidebar}
                    />

                    {/* Mobile overlay — visible only when sidebar is open on small screens */}
                    <div
                        className={`${styles.overlay} ${isSidebarOpen ? styles.overlayVisible : ""}`}
                        onClick={closeSidebar}
                        aria-hidden="true"
                    />

                    <div className={`${styles.sidebar} ${isSidebarOpen ? styles.sidebarOpen : ''}`}>
                        <Sidebar
                            onNewNote={() => navigate('/home')}
                            closeSidebar={closeSidebar}
                        />
                    </div>
                </>
            )}
            <div
                className={styles.mainContent}
                onClick={isSidebarOpen ? closeSidebar : undefined} // close sidebar when clicked note
            >
                <Routes>
                    <Route path="/login" element={<Login />} />
                    <Route path="/register" element={<Register />} />
                    <Route path="/" element={<LandingPage />} />
                    <Route
                        path="/home"
                        element={
                            <ProtectedRoute>
                                <MainContent />
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/review"
                        element={
                            <ProtectedRoute>
                                <ReportPage />
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/note/:id"
                        element={
                            <ProtectedRoute>
                                <NotePage />
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/memory"
                        element={
                            <ProtectedRoute>
                                <MemoryPage />
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/agent"
                        element={
                            <ProtectedRoute>
                                <AgentDemoPage />
                            </ProtectedRoute>
                        }
                    />
                    <Route path="*" element={<Navigate to="/home" replace />} />
                </Routes>
            </div>
        </div>
    );
}

export default AppContent;