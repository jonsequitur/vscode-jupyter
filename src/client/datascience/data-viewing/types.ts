// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
'use strict';

import { IDisposable } from '../../common/types';
import { SharedMessages } from '../messages';

export const CellFetchAllLimit = 100000;
export const CellFetchSizeFirst = 100000;
export const CellFetchSizeSubsequent = 1000000;
export const MaxStringCompare = 200;
export const ColumnWarningSize = 1000; // Anything over this takes too long to load

export namespace DataViewerRowStates {
    export const Fetching = 'fetching';
    export const Skipped = 'skipped';
}

export namespace DataViewerMessages {
    export const Started = SharedMessages.Started;
    export const UpdateSettings = SharedMessages.UpdateSettings;
    export const InitializeData = 'init';
    export const GetAllRowsRequest = 'get_all_rows_request';
    export const GetAllRowsResponse = 'get_all_rows_response';
    export const GetRowsRequest = 'get_rows_request';
    export const GetRowsResponse = 'get_rows_response';
    export const CompletedData = 'complete';
    export const GetCellDetailRequest = 'get_cell_detail_request';
    export const GetCellDetailResponse = 'get_cell_detail_response';
}

export interface IGetRowsRequest {
    start: number;
    end: number;
}

export interface IGetRowsResponse {
    rows: IRowsResponse;
    start: number;
    end: number;
}

// Map all messages to specific payloads
export type IDataViewerMapping = {
    [DataViewerMessages.Started]: never | undefined;
    [DataViewerMessages.UpdateSettings]: string;
    [DataViewerMessages.InitializeData]: IDataFrameInfo;
    [DataViewerMessages.GetAllRowsRequest]: never | undefined;
    [DataViewerMessages.GetAllRowsResponse]: IRowsResponse;
    [DataViewerMessages.GetRowsRequest]: IGetRowsRequest;
    [DataViewerMessages.GetRowsResponse]: IGetRowsResponse;
    [DataViewerMessages.CompletedData]: never | undefined;
    [DataViewerMessages.GetCellDetailRequest]: ISlickGridCellCoordinates;
    [DataViewerMessages.GetCellDetailResponse]: ISlickGridCellDetail;
};

export interface IDataFrameInfo {
    columns?: { key: string; type: ColumnType }[];
    indexColumn?: string;
    rowCount?: number;
    shape?: string;
}

export interface IDataViewerDataProvider {
    dispose(): void;
    getDataFrameInfo(): Promise<IDataFrameInfo>;
    getAllRows(): Promise<IRowsResponse>;
    getRows(start: number, end: number): Promise<IRowsResponse>;
    getDetail?(row: number, column: number): Promise<ISlickGridCellDetail | undefined>;
}

export enum ColumnType {
    String = 'string',
    Number = 'number',
    Bool = 'bool'
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type IRowsResponse = any[];

export const IDataViewerFactory = Symbol('IDataViewerFactory');
export interface IDataViewerFactory {
    create(dataProvider: IDataViewerDataProvider, title: string): Promise<IDataViewer>;
}

export const IDataViewer = Symbol('IDataViewer');
export interface IDataViewer extends IDisposable {
    showData(dataProvider: IDataViewerDataProvider, title: string): Promise<void>;
}

export interface ISlickGridCellCoordinates {
    row: number;
    column: number;
}

export interface ISlickGridCellDetail extends ISlickGridCellCoordinates {
    data: string | undefined;
}
