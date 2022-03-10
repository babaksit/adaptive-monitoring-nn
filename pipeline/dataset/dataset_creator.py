import glob
import logging
import os
from datetime import timedelta, datetime
from pathlib import Path

import dateutil.rrule as rrule
import pandas as pd
from prometheus_api_client import PrometheusConnect
from prometheus_pandas import query
import time
from pipeline.prometheus.handler import PrometheusHandler
import json


class DatasetCreator:
    """
    Create a pandas dataframe by fetching from prometheus

    """

    def __init__(self):
        self.date_format_str = '%Y-%m-%dT%H:%M:%SZ'
        self.merged_csv = None
        self.config = None
        self.prometheus_handler = None
        self.node_exporter_ignore_list = ['node_arp_entries', 'node_boot_time_seconds', 'node_cooling_device_cur_state',
                                          'node_cooling_device_max_state', 'node_cpu_frequency_max_hertz',
                                          'node_cpu_frequency_min_hertz', 'node_cpu_guest_seconds_total',
                                          'node_cpu_scaling_frequency_max_hertz',
                                          'node_cpu_scaling_frequency_min_hertz', 'node_disk_discards_merged_total',
                                          'node_disk_info', 'node_netstat_Ip6_OutOctets', 'node_netstat_Ip_Forwarding',
                                          'node_netstat_TcpExt_ListenDrops', 'node_netstat_TcpExt_ListenOverflows',
                                          'node_netstat_TcpExt_SyncookiesFailed', 'node_netstat_TcpExt_SyncookiesRecv',
                                          'node_netstat_TcpExt_SyncookiesSent', 'node_netstat_TcpExt_TCPSynRetrans',
                                          'node_memory_Bounce_bytes', 'node_memory_CmaFree_bytes',
                                          'node_memory_CmaTotal_bytes', 'node_memory_CommitLimit_bytes',
                                          'node_memory_FileHugePages_bytes', 'node_memory_FilePmdMapped_bytes',
                                          'node_sockstat_UDP6_inuse', 'node_sockstat_UDPLITE6_inuse',
                                          'node_sockstat_UDPLITE_inuse', 'node_sockstat_UDP_inuse',
                                          'node_softnet_dropped_total', 'node_textfile_scrape_error',
                                          'node_timex_estimated_error_seconds', 'node_timex_pps_calibration_total',
                                          'node_timex_pps_error_total', 'node_timex_pps_frequency_hertz',
                                          'node_timex_pps_jitter_seconds', 'node_edac_uncorrectable_errors_total',
                                          'node_hwmon_sensor_label', 'node_memory_HardwareCorrupted_bytes',
                                          'node_netstat_Icmp_OutMsgs', 'node_netstat_Tcp_InErrs',
                                          'node_netstat_UdpLite_InErrors', 'node_network_net_dev_group',
                                          'node_timex_pps_jitter_total', 'node_network_protocol_type',
                                          'node_network_receive_compressed_total', 'node_network_receive_drop_total',
                                          'node_network_receive_errs_total', 'node_network_receive_fifo_total',
                                          'node_network_receive_frame_total', 'node_network_receive_multicast_total',
                                          'node_network_speed_bytes', 'node_network_transmit_carrier_total',
                                          'node_network_transmit_colls_total', 'node_network_transmit_compressed_total',
                                          'node_network_transmit_drop_total', 'node_network_transmit_errs_total',
                                          'node_network_transmit_fifo_total', 'node_network_transmit_queue_length',
                                          'node_network_up', 'node_nf_conntrack_entries_limit', 'node_os_info',
                                          'node_os_version', 'node_memory_ShmemHugePages_bytes',
                                          'node_memory_ShmemPmdMapped_bytes', 'node_memory_SwapCached_bytes',
                                          'node_memory_SwapFree_bytes', 'node_memory_SwapTotal_bytes',
                                          'node_memory_Unevictable_bytes', 'node_memory_VmallocChunk_bytes',
                                          'node_memory_VmallocTotal_bytes', 'node_entropy_pool_size_bits',
                                          'node_exporter_build_info', 'node_filefd_maximum',
                                          'node_filesystem_device_error', 'node_filesystem_files',
                                          'node_filesystem_readonly', 'node_filesystem_size_bytes',
                                          'node_hwmon_chip_names', 'node_pressure_memory_stalled_seconds_total',
                                          'node_pressure_memory_waiting_seconds_total', 'node_scrape_collector_success',
                                          'node_sockstat_FRAG6_inuse', 'node_sockstat_FRAG6_memory',
                                          'node_sockstat_FRAG_inuse', 'node_sockstat_FRAG_memory',
                                          'node_timex_pps_shift_seconds', 'node_timex_pps_stability_exceeded_total',
                                          'node_timex_pps_stability_hertz', 'node_timex_sync_status',
                                          'node_timex_tai_offset_seconds', 'node_timex_tick_seconds',
                                          'node_time_clocksource_available_info', 'node_time_clocksource_current_info',
                                          'node_time_zone_offset_seconds', 'node_udp_queues', 'node_uname_info',
                                          'node_vmstat_oom_kill', 'node_vmstat_pswpin', 'node_vmstat_pswpout',
                                          'node_memory_Hugepagesize_bytes', 'node_memory_HugePages_Free',
                                          'node_memory_HugePages_Rsvd', 'node_memory_HugePages_Surp',
                                          'node_memory_HugePages_Total', 'node_memory_Hugetlb_bytes',
                                          'node_memory_MemTotal_bytes', 'node_memory_Mlocked_bytes',
                                          'node_memory_NFS_Unstable_bytes', 'node_netstat_Udp_InDatagrams',
                                          'node_netstat_Udp_InErrors', 'node_netstat_Udp_NoPorts',
                                          'node_netstat_Udp_OutDatagrams', 'node_netstat_Udp_RcvbufErrors',
                                          'node_netstat_Udp_SndbufErrors', 'node_network_address_assign_type',
                                          'node_network_carrier', 'node_network_carrier_changes_total',
                                          'node_network_carrier_down_changes_total',
                                          'node_network_carrier_up_changes_total', 'node_network_device_id',
                                          'node_network_dormant', 'node_network_flags', 'node_network_iface_id',
                                          'node_network_iface_link', 'node_network_iface_link_mode',
                                          'node_network_info', 'node_network_mtu_bytes',
                                          'node_network_name_assign_type', 'node_dmi_info',
                                          'node_edac_correctable_errors_total',
                                          'node_edac_csrow_correctable_errors_total',
                                          'node_edac_csrow_uncorrectable_errors_total', 'node_hwmon_temp_max_celsius',
                                          'node_ipvs_connections_total', 'node_ipvs_incoming_bytes_total',
                                          'node_ipvs_incoming_packets_total', 'node_ipvs_outgoing_bytes_total',
                                          'node_ipvs_outgoing_packets_total', 'node_md_blocks', 'node_md_blocks_synced',
                                          'node_md_disks', 'node_md_disks_required', 'node_md_state',
                                          'node_netstat_Udp6_InDatagrams', 'node_netstat_Udp6_InErrors',
                                          'node_netstat_Udp6_NoPorts', 'node_netstat_Udp6_OutDatagrams',
                                          'node_netstat_Udp6_RcvbufErrors', 'node_netstat_Udp6_SndbufErrors',
                                          'node_netstat_UdpLite6_InErrors', 'node_memory_WritebackTmp_bytes',
                                          'node_namespace_pod:kube_pod_info:', 'node_netstat_Icmp6_InErrors',
                                          'node_netstat_Icmp6_InMsgs', 'node_netstat_Icmp6_OutMsgs',
                                          'node_netstat_Icmp_InErrors', 'node_netstat_Icmp_InMsgs']
        self.node_exporter_counters = ["node_context_switches_total", "node_cpu_guest_seconds_total",
                                       "node_cpu_seconds_total", "node_disk_discard_time_seconds_total",
                                       "node_disk_discarded_sectors_total", "node_disk_discards_completed_total",
                                       "node_disk_discards_merged_total", "node_disk_io_time_seconds_total",
                                       "node_disk_io_time_weighted_seconds_total", "node_disk_read_bytes_total",
                                       "node_disk_read_time_seconds_total", "node_disk_reads_completed_total",
                                       "node_disk_reads_merged_total", "node_disk_write_time_seconds_total",
                                       "node_disk_writes_completed_total", "node_disk_writes_merged_total",
                                       "node_disk_written_bytes_total", "node_edac_correctable_errors_total",
                                       "node_edac_csrow_correctable_errors_total",
                                       "node_edac_csrow_uncorrectable_errors_total",
                                       "node_edac_uncorrectable_errors_total", "node_forks_total", "node_intr_total",
                                       "node_ipvs_connections_total", "node_ipvs_incoming_bytes_total",
                                       "node_ipvs_incoming_packets_total", "node_ipvs_outgoing_bytes_total",
                                       "node_ipvs_outgoing_packets_total", "node_network_carrier_changes_total",
                                       "node_network_carrier_down_changes_total",
                                       "node_network_carrier_up_changes_total", "node_network_receive_bytes_total",
                                       "node_network_receive_compressed_total", "node_network_receive_drop_total",
                                       "node_network_receive_errs_total", "node_network_receive_fifo_total",
                                       "node_network_receive_frame_total", "node_network_receive_multicast_total",
                                       "node_network_receive_packets_total", "node_network_transmit_bytes_total",
                                       "node_network_transmit_carrier_total", "node_network_transmit_colls_total",
                                       "node_network_transmit_compressed_total", "node_network_transmit_drop_total",
                                       "node_network_transmit_errs_total", "node_network_transmit_fifo_total",
                                       "node_network_transmit_packets_total", "node_pressure_cpu_waiting_seconds_total",
                                       "node_pressure_io_stalled_seconds_total",
                                       "node_pressure_io_waiting_seconds_total",
                                       "node_pressure_memory_stalled_seconds_total",
                                       "node_pressure_memory_waiting_seconds_total",
                                       "node_schedstat_running_seconds_total", "node_schedstat_timeslices_total",
                                       "node_schedstat_waiting_seconds_total", "node_softnet_dropped_total",
                                       "node_softnet_processed_total", "node_softnet_times_squeezed_total",
                                       "node_timex_pps_calibration_total", "node_timex_pps_error_total",
                                       "node_timex_pps_jitter_total", "node_timex_pps_stability_exceeded_total",
                                       "process_cpu_seconds_total", "promhttp_metric_handler_errors_total",
                                       "promhttp_metric_handler_requests_total"]

    def create_prometheus_queries_df(self, config_path: str, save: bool = True) -> pd.DataFrame:

        with open(config_path) as f:
            self.config = json.load(f)

        start_time_str = self.config["start_time"]
        end_time_str = self.config["end_time"]
        save_dir = self.config["save_dir"]
        step = self.config["step"]
        prometheus_url = self.config["prometheus_url"]
        queries = [query['query'] for query in self.config["queries"]]
        columns = [query['column_name'] for query in self.config["queries"]]

        if len(columns) != len(queries):
            raise ValueError("column names should have same size as queries, check config!")
            return None

        self.prometheus_handler = PrometheusHandler(prometheus_url)

        # prometheus unique save dir
        save_time = str(time.ctime()) + ".csv"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        # final save dir
        save_path = os.path.join(save_dir, save_time)

        df = self.prometheus_handler.fetch_queries(start_time_str, end_time_str,
                                                   queries, columns, step)
        if save:
            df.to_csv(save_path)
        return df


if __name__ == '__main__':
    logging.basicConfig(filename='dataset_creator.log', level=logging.DEBUG)
    dc = DatasetCreator()
    # dc.create_prometheus_df(start_time_str="2022-02-18T11:30:00Z",
    #                         end_time_str="2022-02-25T13:00:00Z",
    #                         start_metric_name="node_",
    #                         save_dir="/home/mounted_drive/thesis/prometheus",
    #                         use_rate=False,
    #                         step="1s")
    # dc.merge_prometheus_dfs("data/prometheus/Mon Feb 28 13:37:24 2022")
    dc.create_prometheus_queries_df("/home/bsi/adaptive-monitoring-nn/pipeline/configs/dataset.json")
