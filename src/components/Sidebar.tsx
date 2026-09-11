import { useCallback, useMemo, useRef } from "react";
import styles from "../styles/Sidebar.module.scss";
import { useNavigate } from "react-router-dom";
import api from "../api/axios"
import { handleAuthExpired } from "../utils/handleAuthExpired";
import {
    useNotesInfiniteQuery,
    NoteListItem,
} from "../hooks/useNotesInfiniteQuery";
import SidebarNoteItem from "./SidebarNoteItem";

interface SidebarProps {
    onNewNote: () => void;
    // onSelectNote: (id: string) => void;
    closeSidebar?: () => void;
}

const Sidebar = ({ closeSidebar, onNewNote }: SidebarProps) => {
    const {
        data,
        fetchNextPage,
        hasNextPage,
        isFetchingNextPage,
        isLoading,
    } = useNotesInfiniteQuery();


    const observer = useRef<IntersectionObserver | null>(null);
    const lastNoteRef = useCallback(
        (node: HTMLLIElement | null) => {
            if (isFetchingNextPage) return
            if (observer.current) observer.current.disconnect()
            observer.current = new IntersectionObserver((entries) => {
                if (entries[0].isIntersecting && hasNextPage) {
                    fetchNextPage()
                }
            })
            if (node) observer.current.observe(node)
        },
        [isFetchingNextPage, hasNextPage, fetchNextPage]
    )
    const navigate = useNavigate();


    const handleLogout = async () => {
        try {
            await api.post('/auth/logout', {}, {
                headers: { 'X-Skip-Retry': 'true' }
            });
        } catch (err: any) {
            console.error('Logout failed:', err);
        } finally {
            if (closeSidebar) closeSidebar();
            handleAuthExpired()
            navigate('/login', { replace: true });
        }
    };

    const { groupedNotes, sortedDates } = useMemo(() => {
        const allNotes: NoteListItem[] = data?.pages.flatMap((p) => p.notes) ?? [];
        const grouped = allNotes.reduce((acc, note) => {
            const date = new Date(note.date).toLocaleDateString();
            if (!acc[date]) acc[date] = [];
            acc[date].push(note);
            return acc;
        }, {} as Record<string, NoteListItem[]>);

        const sorted = Object.keys(grouped).sort(
            (a, b) => new Date(b).getTime() - new Date(a).getTime()
        );

        return { groupedNotes: grouped, sortedDates: sorted };
    }, [data]);



    const handleNoteSelect = (id: string) => {
        navigate(`/note/${id}`);
        closeSidebar?.();
    };

    return (
        <div className={styles.sidebar}>
            <div className={styles.header}>
                <h2>Notes</h2>

                <button className={styles.logoutButton} onClick={handleLogout}>
                    Logout
                </button>

                <button
                    className={styles.summaryButton}
                    onClick={
                        () => {
                            navigate("/review")
                            if (closeSidebar) closeSidebar()
                        }}
                >
                    Letters
                </button>
                <button
                    className={styles.summaryButton}
                    onClick={() => {
                        navigate("/memory")
                        if (closeSidebar) closeSidebar()
                    }}
                >
                    Memory
                </button>
                <button
                    className={styles.summaryButton}
                    onClick={() => {
                        navigate("/agent")
                        if (closeSidebar) closeSidebar()
                    }}
                >
                    Agent
                </button>
                <button className={styles.newNoteButton} onClick={() => {
                    onNewNote()
                    if (closeSidebar) closeSidebar()
                }}
                    aria-label="Create new note"
                >
                    + New Note
                </button>
            </div>
            <div className={styles.noteGroups}>
                {isLoading ? (
                    <p className={styles.empty}>Loading notes...</p>
                ) : sortedDates.length === 0 ? (
                    <p className={styles.empty}>No History</p>
                ) : (sortedDates.map((date) => (
                    <div key={date} className={styles.dateGroup}>
                        <h3 className={styles.dateTitle}>{date}</h3>
                        <ul className={styles.noteList}>
                            {groupedNotes[date].map((note, index) => {
                                const isLastInGroup =
                                    index === groupedNotes[date].length - 1 &&
                                    date === sortedDates[sortedDates.length - 1]

                                return (
                                    <li
                                        key={note.id}
                                        ref={isLastInGroup ? lastNoteRef : null}
                                        className={styles.noteItem}
                                        onClick={() => handleNoteSelect(note.id)}
                                        role="button"
                                        tabIndex={0}
                                        onKeyDown={(e) => {
                                            if (e.key === "Enter" || e.key === " ") {
                                                handleNoteSelect(note.id);
                                            }
                                        }}
                                        aria-label={`Select note: ${note.title}`}
                                    >
                                        <SidebarNoteItem note={note} />
                                    </li>
                                )
                            })}

                        </ul>
                    </div>)
                )
                )}
                {isFetchingNextPage && <p className={styles.loadingMore}>Loading more...</p>}
            </div>
        </div >
    );
};

export default Sidebar;