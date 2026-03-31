from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.services.ingestion_service import ingest_folder
from src.services.library_service import rebuild_index_from_db, scan_indexed_library
from src.services.search_service import search_by_text


st.set_page_config(
    page_title='Prompt2PictureAgent',
    page_icon='🖼️',
    layout='wide',
)


def render_metrics(title: str, items: list[dict]) -> None:
    st.metric(label=title, value=len(items))


def render_result_grid(results: list[dict], columns_count: int = 3) -> None:
    if not results:
        st.info('No results to display.')
        return

    columns = st.columns(columns_count)

    for idx, result in enumerate(results):
        column = columns[idx % columns_count]

        with column:
            image_path = result['path']
            st.markdown(f"**Rank {result['rank']}**")
            st.caption(result['filename'])

            try:
                image = Image.open(image_path).convert('RGB')
                st.image(image, use_container_width=True)
            except Exception as e:
                st.warning(f'Could not open image: {e}')

            st.write(f"Similarity: {result['similarity_score']:.4f}")
            st.write(f"Raw distance: {result['raw_distance']:.4f}")
            st.code(image_path, language='text')


def render_report_section(title: str, items: list[dict]) -> None:
    st.subheader(title)
    st.write(f'Count: {len(items)}')

    if items:
        st.dataframe(items, use_container_width=True)
    else:
        st.caption('No items.')


def search_tab() -> None:
    st.header('Semantic Search')

    with st.form('search_form'):
        query = st.text_input('Search prompt', placeholder='a dog running in the grass')
        top_k = st.slider('Top K results', min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button('Search')

    if submitted:
        try:
            results = search_by_text(query=query, top_k=top_k)
            st.session_state['search_results'] = results
        except Exception as e:
            st.error(f'Search failed: {e}')

    results = st.session_state.get('search_results', [])
    if results:
        render_result_grid(results)


def ingestion_tab() -> None:
    st.header('Add Folder to Library')
    st.caption('This app uses linked local files. It indexes images in place and does not copy them.')

    with st.form('ingestion_form'):
        folder_path = st.text_input(
            'Folder path',
            placeholder='/home/cyrus/Pictures/trip_photos',
        )
        recursive = st.checkbox('Scan subfolders recursively', value=True)
        batch_size = st.slider('Embedding batch size', min_value=1, max_value=128, value=16)
        submitted = st.form_submit_button('Ingest Folder')

    if submitted:
        try:
            report = ingest_folder(
                folder_path=folder_path,
                recursive=recursive,
                batch_size=batch_size,
            )
            st.session_state['ingestion_report'] = report
            st.success('Folder ingestion completed.')
        except Exception as e:
            st.error(f'Ingestion failed: {e}')

    report = st.session_state.get('ingestion_report')
    if report:
        st.subheader('Ingestion Summary')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Discovered', report.get('discovered_count', 0))
        with col2:
            st.metric('Ingested', len(report.get('ingested', [])))
        with col3:
            st.metric('Skipped', len(report.get('skipped', [])))
        with col4:
            st.metric('Failed', len(report.get('failed', [])))

        with st.expander('Ingested', expanded=True):
            render_report_section('Ingested files', report.get('ingested', []))

        with st.expander('Skipped'):
            render_report_section('Skipped files', report.get('skipped', []))

        with st.expander('Failed'):
            render_report_section('Failed files', report.get('failed', []))


def library_tab() -> None:
    st.header('Library Health')

    col1, col2 = st.columns(2)

    with col1:
        if st.button('Scan Library', use_container_width=True):
            try:
                report = scan_indexed_library()
                st.session_state['scan_report'] = report
                st.success('Library scan completed.')
            except Exception as e:
                st.error(f'Scan failed: {e}')

    with col2:
        if st.button('Rebuild Index', use_container_width=True):
            try:
                report = rebuild_index_from_db(batch_size=16)
                st.session_state['rebuild_report'] = report
                st.success('Index rebuild completed.')
            except Exception as e:
                st.error(f'Rebuild failed: {e}')

    scan_report = st.session_state.get('scan_report')
    if scan_report:
        st.subheader('Scan Summary')
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metrics('Available', scan_report.get('available', []))
        with col2:
            render_metrics('Missing', scan_report.get('missing', []))
        with col3:
            render_metrics('Invalid', scan_report.get('invalid', []))

        with st.expander('Available'):
            render_report_section('Available records', scan_report.get('available', []))

        with st.expander('Missing', expanded=True):
            render_report_section('Missing records', scan_report.get('missing', []))

        with st.expander('Invalid'):
            render_report_section('Invalid records', scan_report.get('invalid', []))

    rebuild_report = st.session_state.get('rebuild_report')
    if rebuild_report:
        st.subheader('Rebuild Summary')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Reindexed', len(rebuild_report.get('reindexed', [])))
        with col2:
            st.metric('Missing', len(rebuild_report.get('missing', [])))
        with col3:
            st.metric('Failed', len(rebuild_report.get('failed', [])))

        with st.expander('Reindexed', expanded=True):
            render_report_section('Reindexed records', rebuild_report.get('reindexed', []))

        with st.expander('Missing'):
            render_report_section('Missing records', rebuild_report.get('missing', []))

        with st.expander('Failed'):
            render_report_section('Failed records', rebuild_report.get('failed', []))


def main() -> None:
    st.title('Prompt2PictureAgent')
    st.caption('A local semantic photo search app for linked personal image libraries.')

    search, ingest, library = st.tabs(['Search', 'Ingest', 'Library'])

    with search:
        search_tab()

    with ingest:
        ingestion_tab()

    with library:
        library_tab()


if __name__ == '__main__':
    main()